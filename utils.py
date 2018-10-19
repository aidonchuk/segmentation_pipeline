import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x # async=True


def write_event(log, step, epoch, **data):
    data['step'] = step
    data['epoch'] = epoch
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, optimizer, scheduler, n_epochs=None,
          fold=None):
    n_epochs = n_epochs or args.n_epochs
    root = Path(args.root + "/" + args.model)
    model_path = root / 'fold_{fold}/{start_epoch}_model_{fold}.pt'.format(fold=fold, start_epoch=args.start_epoch)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 0
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(str(root) + '/fold_{fold}/'.format(fold=fold) + str(ep) + '_model_{fold}.pt'.format(fold=fold)))

    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    valid_metric = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        scheduler.step(epoch)
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0].get('lr')))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss_mean = criterion(outputs, targets)
                # loss_mean, loss_arr = hem(loss_mean, loss_arr, inputs, targets, model, criterion,
                #                          sample_count=args.hem_sample_count)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss_mean.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss_mean.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, epoch, loss=mean_loss)
            write_event(log, step, epoch, loss=mean_loss)
            tq.close()
            save(epoch)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, epoch, **valid_metrics)
            valid_losses.append(valid_metrics['valid_loss'])
            valid_metric.append(float(valid_metrics['kaggel_metric']))
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return

        if early_stop(valid_losses, args.early_stop_patience):
            print('Early stopping.')
            break

    rm_all_but_5best_and_last(valid_metric, fold, root, args.start_epoch, args)


def rm_all_but_5best_and_last(valid_metric, fold, root, start_epoch, args):
    valid_metric = np.asarray(valid_metric, np.float64)
    valid_metric = valid_metric[np.arange(valid_metric.size - 1)]
    ids = np.argsort(valid_metric)
    ids = ids[:len(ids) - args.save_best_count]
    for i in ids:
        os.remove(
            str(str(root) + '/fold_{fold}/'.format(fold=fold) + str(i + int(start_epoch)) + '_model_{fold}.pt'.format(
                fold=fold)))


def hem(loss_mean, loss_arr, inputs, targets, model, criterion, sample_count=0):
    if sample_count > 0:
        idx = np.argsort(loss_arr)
        # loss_arr = loss_arr[idx][-samples_num:]
        inputs = inputs[idx][-sample_count:]
        targets = targets[idx][-sample_count:]

        outputs = model(inputs)
        m, a = criterion(outputs, targets)
        loss_mean = (m + loss_mean) / 2
        return loss_mean, loss_arr
    else:
        return loss_mean, loss_arr


def early_stop(valid_losses, patience):
    index = np.argmin(valid_losses)
    if len(valid_losses) - index > patience:
        return True
    return False
