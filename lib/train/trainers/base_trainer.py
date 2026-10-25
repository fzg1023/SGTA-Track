import os
import glob
import gc
import subprocess
import sys
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler


def env_int(name, default):
    """Read an int from the environment, tolerating unset/blank/invalid values.

    ``export VAR=`` (the usual idiom for an optional override) exports an empty
    string, which ``os.environ.get(name, default)`` would return verbatim and
    ``int()`` would then reject.  Blank therefore means "not set".
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        print('[WARN] ignoring invalid %s=%r, using %r' % (name, raw, default))
        return default


def offload_training_state(net, optimizer=None):
    """Park the training model (and optimizer state) in host memory.

    The per-epoch evaluation subprocess runs on the same device while the
    training process keeps its model and optimizer alive.  Moving them to CPU
    before the test removes the OOM risk entirely, and costs nothing because no
    training happens while the test runs.

    Returns a token that must be handed back to ``restore_training_state``.
    """
    params = list(net.parameters())
    token = {
        'model_device': params[0].device if params else None,
        'optimizer_devices': [],
    }
    if token['model_device'] is not None and token['model_device'].type == 'cpu':
        token['model_device'] = None  # already parked, nothing to do
    else:
        net.cpu()

    if optimizer is not None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    token['optimizer_devices'].append(value.device)
                    state[key] = value.cpu()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return token


def restore_training_state(net, optimizer, token):
    """Undo ``offload_training_state`` (call it even when the test failed)."""
    if token is None:
        return
    device = token.get('model_device')
    if device is not None and device.type != 'cpu':
        net.to(device)

    if optimizer is not None:
        devices = token.get('optimizer_devices') or []
        index = 0
        for state in optimizer.state.values():
            for key, value in state.items():
                if not torch.is_tensor(value):
                    continue
                if index < len(devices):
                    target = devices[index]
                    if target.type != 'cpu':
                        state[key] = value.to(target)
                index += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir''' # todo: assure usage
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
            _checkpoint_dir - output/checkpoints
        """

        epoch = -1
        # Retry instead of stopping silently. A CUDA OOM late in a long ablation
        # run should resume from the last saved checkpoint rather than end the
        # job; epochs skipped while recovering can be filled in afterwards with
        # the evaluation entry point, e.g.
        #   python RGBT_workspace/test_sgta.py --yaml_name lasher_train --epoch 12
        num_tries = max(1, env_int('TRAIN_RETRIES', 3))
        for i in range(num_tries):
            try:
                latest_loaded = False
                if load_latest:
                    latest_loaded = bool(self.load_checkpoint())
                if load_previous_ckpt and not latest_loaded:
                    directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                    self.load_state_dict(directory)
                if distill:
                    directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                    self.load_state_dict(directory_teacher, distill=True)
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    # only save the last 10 checkpoints
                    save_epoch_interval = getattr(self.settings, "save_epoch_interval", 1)
                    save_last_n_epoch = getattr(self.settings, "save_last_n_epoch", 1)
                    if epoch > (max_epochs - save_last_n_epoch) or epoch % save_epoch_interval == 0 or epoch >4:
                        if self._checkpoint_dir:
                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()
                                self._run_epoch_eval()
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')
        # fail_safe swallows the exception (no exit code, and this line prints even
        # after a crash), so state explicitly whether the target was reached --
        # otherwise a run that died at epoch 5 of 30 looks like a success.
        if self.epoch < max_epochs:
            print('[TRAIN] NOTE: last epoch actually run = %d, target = %d.  A crash was '
                  'caught in this run (fail_safe=True, TRAIN_RETRIES=%d) -- search the log '
                  'above for the traceback; re-running with RESUME=1 continues from the '
                  'latest checkpoint.' % (self.epoch, max_epochs, num_tries), flush=True)

    def train_epoch(self):
        raise NotImplementedError

    def _run_epoch_eval(self):
        """Hook called on rank 0 right after the epoch checkpoint is saved.

        When cfg.TRAIN.EVAL_LASHER_EVERY_EPOCH is enabled, launches a subprocess
        that runs a complete lasher testingset evaluation (tracking + PR/NPR/SR)
        with the checkpoint that was just saved. Other ranks wait on a
        distributed barrier while the test runs.
        """
        cfg = getattr(self.settings, 'cfg', None)
        if cfg is None or not getattr(cfg.TRAIN, 'EVAL_LASHER_EVERY_EPOCH', False):
            return

        # Checkpoints are saved every epoch; the full benchmark can be throttled
        # to every N epochs with TRAIN.EVAL_LASHER_EPOCH_INTERVAL (env override
        # EVAL_LASHER_EPOCH_INTERVAL).  Default 1 keeps the old behaviour.
        eval_interval = int(os.environ.get(
            'EVAL_LASHER_EPOCH_INTERVAL',
            getattr(cfg.TRAIN, 'EVAL_LASHER_EPOCH_INTERVAL', 1) or 1))
        if eval_interval > 1 and self.epoch % eval_interval != 0:
            print('[EPOCH_EVAL] skipped at epoch {} (interval {})'.format(
                self.epoch, eval_interval), flush=True)
            return

        # Which benchmark the per-epoch hook evaluates. LasHeR by default so that
        # existing configs keep working; set TRAIN.EVAL_DATASET (or the
        # EVAL_DATASET env var) to e.g. "vtuav_st" / "vtuav_lt".
        eval_dataset = os.environ.get(
            'EVAL_DATASET', getattr(cfg.TRAIN, 'EVAL_DATASET', 'lasher') or 'lasher')

        try:
            import torch.distributed as dist
        except Exception:
            dist = None
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.barrier()  # wait so all ranks are idle while the test runs

        try:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
            net_type = type(net).__name__
            checkpoint = os.path.join(self._checkpoint_dir, self.settings.project_path,
                                      '{}_ep{:04d}.pth.tar'.format(net_type, self.epoch))
            # cfg_file = <prj>/experiments/<script>/<config>.yaml -> go up 3 levels to <prj>
            prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(self.settings.cfg_file))))
            test_script = os.path.join(prj_dir, 'RGBT_workspace', 'test_sgta.py')
            if not os.path.isfile(test_script):
                print('[EPOCH_EVAL] test script not found, skip: %s' % test_script)
                return

            workers = env_int('EVAL_LASHER_WORKERS',
                              getattr(cfg.TRAIN, 'EVAL_LASHER_WORKERS', 4))
            eval_gpu = os.environ.get('EVAL_LASHER_GPU', getattr(cfg.TRAIN, 'EVAL_LASHER_GPU', '') or '')
            save_dir = os.environ.get('EVAL_LASHER_SAVE_DIR',
                                      getattr(cfg.TRAIN, 'EVAL_LASHER_SAVE_DIR', './RGBT_workspace/results'))
            if not os.path.isabs(save_dir):
                save_dir = os.path.join(prj_dir, save_dir)

            # Optional results namespace: separates an ablation row's dumps and
            # its own eval_history.csv from the main results tree, e.g.
            #   <save_dir>/lasher_a0/<config>_<epoch>/
            # Empty -> the dataset name is used (legacy layout).
            results_ns = os.environ.get(
                'EVAL_LASHER_RESULTS_NS',
                getattr(cfg.TRAIN, 'EVAL_LASHER_RESULTS_NS', '') or '').strip() or eval_dataset

            cmd = [sys.executable, test_script,
                   '--checkpoint', checkpoint,
                   '--dataset', eval_dataset,
                   '--yaml_name', self.settings.config_name,
                   '--epoch', str(self.epoch),
                   '--workers', str(workers),
                   '--save_dir', save_dir,
                   '--results_ns', results_ns]
            # Diagnostic shortcut: evaluate a single sequence so the whole
            # save -> offload -> test -> restore chain can be smoke-tested in
            # seconds instead of hours.  Unset by default.
            eval_sequence = os.environ.get('EVAL_LASHER_SEQUENCE', '').strip()
            if eval_sequence:
                cmd += ['--sequence', eval_sequence]
                print('[EPOCH_EVAL] single-sequence mode: %s' % eval_sequence, flush=True)
            env = dict(os.environ)
            # unbuffered child stdout: the per-sequence results must show up in
            # the training terminal as they finish, not in 4 KiB blocks
            env['PYTHONUNBUFFERED'] = '1'
            if eval_gpu:
                env['CUDA_VISIBLE_DEVICES'] = eval_gpu

            # The training model and optimizer stay alive while the evaluation
            # subprocess runs, and each test worker loads its own copy of the
            # network.  Park the training state in host memory first so the
            # device cannot run out of memory during the test; nothing is being
            # trained while the test runs, so this is free.
            token = offload_training_state(net, self.optimizer)
            if token and token.get('model_device') is not None:
                print('[EPOCH_EVAL] training model parked in CPU memory '
                      '(%s -> cpu)' % token['model_device'], flush=True)

            proc = None
            try:
                print('[EPOCH_EVAL] epoch %d: running full %s test ...' % (self.epoch, eval_dataset),
                      flush=True)
                print('[EPOCH_EVAL] cmd: %s' % ' '.join(cmd), flush=True)
                # Output streams directly to the training terminal (the test can
                # take hours, progress must stay visible).
                proc = subprocess.run(cmd, env=env, cwd=prj_dir)
            finally:
                restore_training_state(net, self.optimizer, token)

            if proc is not None and proc.returncode != 0:
                print('[EPOCH_EVAL] WARNING: evaluation for epoch %d exited with code %d; '
                      'its eval_history row may be missing. Re-run '
                      'python RGBT_workspace/test_sgta.py --yaml_name %s --epoch %d to fill it in.'
                      % (self.epoch, proc.returncode, self.settings.config_name, self.epoch),
                      flush=True)

            # Read the metrics row written by test_sgta.py and record it.
            history_csv = os.path.join(save_dir, results_ns, 'eval_history.csv')
            tag = '%s_%d' % (self.settings.config_name, self.epoch)
            if os.path.isfile(history_csv):
                import csv
                with open(history_csv, newline='') as f:
                    rows = list(csv.reader(f))
                header = rows[0] if rows else []

                def col(name, fallback):
                    if header and name in header:
                        return header.index(name)
                    return fallback

                # new rich header uses seq_*; legacy header was [tag, PR, NPR, SR]
                idx_pr = col('seq_PR', 1)
                idx_npr = col('seq_NPS', 2)
                idx_sr = col('seq_SS', 3)
                # Official LasHeR trio written by test_sgta.py.  It uses the
                # official protocol (NPR threshold 0.2, threshold-averaged SR,
                # (w-1)/2 centre, rounded boxes, frame 0 forced to GT) and is
                # verified against rgbt 1.0.1 by tools/check_official_metric_parity.py
                idx_opr = col('seq_oPR', -1)
                idx_onpr = col('seq_oNPR', -1)
                idx_osr = col('seq_oSR', -1)
                for row in rows[1:]:
                    if row and row[0] == tag and len(row) > max(idx_pr, idx_npr, idx_sr):
                        line = '[%s_RESULT] tag=%s PR=%s NPR=%s SR=%s' % (
                            eval_dataset.upper(), tag, row[idx_pr], row[idx_npr], row[idx_sr])
                        print(line)
                        if getattr(self.settings, 'log_file', None):
                            with open(self.settings.log_file, 'a') as f_log:
                                f_log.write(line + '\n')
                        if min(idx_opr, idx_onpr, idx_osr) >= 0 and \
                                len(row) > max(idx_opr, idx_onpr, idx_osr) and \
                                all(row[i] not in ('', None) for i in (idx_opr, idx_onpr, idx_osr)):
                            pct = [100.0 * float(row[i]) for i in (idx_opr, idx_onpr, idx_osr)]
                            line2 = ('[%s_OFFICIAL] tag=%s PR=%.2f NPR=%.2f SR=%.2f'
                                     % (eval_dataset.upper(), tag, pct[0], pct[1], pct[2]))
                            print(line2, flush=True)
                            if getattr(self.settings, 'log_file', None):
                                with open(self.settings.log_file, 'a') as f_log:
                                    f_log.write(line2 + '\n')
                        break

            # Periodic independent cross-check: recompute the just-written dump
            # directory with the official toolkit and compare.  Cheap (~15 s) and
            # catches any drift in the in-process implementation.  Skipped for a
            # single-sequence dump: the toolkit iterates over every sequence of
            # the dataset, so the check can only run on a complete evaluation.
            if eval_sequence:
                pass
            elif str(os.environ.get('EVAL_OFFICIAL_TOOLKIT', '1')).lower() in ('1', 'true', 'yes', 'on'):
                every = env_int('EVAL_OFFICIAL_TOOLKIT_EVERY', 5)
                if every > 0 and self.epoch % every == 0:
                    chk = os.path.join(prj_dir, 'tools', 'check_official_metric_parity.py')
                    if os.path.isfile(chk):
                        result_dir = os.path.join(save_dir, results_ns,
                                                  '%s_%d' % (self.settings.config_name, self.epoch))
                        if os.path.isdir(result_dir):
                            try:
                                out = subprocess.run([sys.executable, chk, '--result-dir', result_dir],
                                                     cwd=prj_dir, capture_output=True, text=True,
                                                     timeout=1800)
                                tail = [ln for ln in (out.stdout or '').splitlines()
                                        if 'PARITY' in ln or ln.strip().startswith(('PR ', 'NPR ', 'SR '))]
                                if not tail:
                                    err = (out.stderr or '').strip().splitlines()
                                    tail = ['rc=%d: %s' % (out.returncode, err[-1] if err else 'no output')]
                                print('[EPOCH_EVAL][TOOLKIT_CHECK] %s' % ' | '.join(tail), flush=True)
                            except Exception as exc:                      # noqa: BLE001
                                print('[EPOCH_EVAL][TOOLKIT_CHECK] skipped: %s' % exc, flush=True)

            if proc.returncode != 0:
                print('[EPOCH_EVAL] WARNING: %s test exited with code %d (training continues)'
                      % (eval_dataset, proc.returncode))
        except Exception:
            print('[EPOCH_EVAL] WARNING: %s test failed (training continues):' % eval_dataset)
            print(traceback.format_exc())
        finally:
            if dist is not None and dist.is_available() and dist.is_initialized():
                dist.barrier()


    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
        # 2021.1.10 Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None, distill=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        if distill:
            net = self.actor.net_teacher.module if multigpu.is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
