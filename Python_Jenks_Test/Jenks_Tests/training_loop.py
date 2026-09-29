from custom_optimizer import Prune_Score_v3, train_one_step_prune_v2, train_one_step_prune, Prune_Score, Prune_Score_v2, train_one_step_prune_global, Prune_Score_Global, train_one_step_prune_v2_ResNet, train_one_step_prune_v2_ResNetETF, train_one_step_prune_v2_ETF, train_one_step_prune_HPO,Prune_Score_Reset
from time import time
from cuda_helpers import get_memory_free_MiB
import torch
import torch.nn as nn
from torchvision.transforms.v2 import CutMix, MixUp
import inspect



def train_val_loop_ResNet_scheduler_ETF(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, warmup_epochs_2=1, min_epochs=1):
    print("ETF babay")
    no_jenks =True
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch == prune_epoch+1:
            print("Changing the learning rate and momentum")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-4
                param_group['momentum'] = 0.98
        if epoch>=prune_epoch and epoch % prune_between == 0:
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            elif not one_shot and epoch>=prune_epoch and epoch % prune_between == 0:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.75
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2_ResNetETF(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs_2,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X,y,training=False)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                # optimizer.zero_grad()
                # with backpack(DiagHessian(), HMP()):
                # # keep graph for autodiff HVPs
                #     loss.backward()
                # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
                # with open(trace_val_filename,"a") as f:
                #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)
        scheduler.step(epoch = epoch, metric = val_acc)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    total_params = 0
    nonzero_params = 0

    for name, param in model.named_parameters():
        module = dict(model.named_modules()).get(name.rsplit('.', 1)[0], None)
        if module is not None and hasattr(module, 'do_prune') and module.do_prune:
            total_params += param.numel()
            nonzero_params += (param != 0).sum().item()
    sparsity = 1 - nonzero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)


def train_val_loop(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, min_epochs=1, elem_bias = False, accum_steps=1, weight_reset=False):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    max_val_acc = 0.0
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch == prune_epoch or (epoch>prune_epoch and (epoch-prune_epoch) % prune_between==0):
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if not weight_reset:
                if one_shot and epoch==prune_epoch:
                    print("Pruning the weights")
                    Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                    prune_count += 1
                elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
                    print("Pruning the weights")
                    Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                    prune_count += 1
            else:
                if one_shot and epoch==prune_epoch:
                    print("Pruning the weights with weight reset")
                    Prune_Score_Reset(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename, weight_reset=True)
                    prune_count += 1
                else:
                    print("Not done or needed to be tested")
                    return
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename, mag=False, elem_bias=elem_bias, accumulation_steps=accum_steps)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        scheduler.step()
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)
            if val_acc/count_val > max_val_acc and epoch>prune_epoch:
                max_val_acc = val_acc/count_val
                torch.save(model.state_dict(), f"models/best_{timestamp}_{experiment_name}_{model_name}.pth")
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)



    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
    with open(val_filename,"a") as f:
        print(f"Best validation accuracy achieved: {max_val_acc: .5f}", file=f)



    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
    with open(val_filename,"a") as f:
        print(f"Best validation accuracy achieved: {max_val_acc: .5f}", file=f)


def train_val_loopETF(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, min_epochs=1):
    no_jenks =False
    l2 = True
    mag_prune = True
    AGD = False
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        weight_decay = [group.get("weight_decay", 0.0) for group in optimizer.param_groups]
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
            print(f"Weight Decay: {weight_decay}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch == prune_epoch or (epoch>prune_epoch and (epoch-prune_epoch) % prune_between==0):
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2_ETF(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename, scheduler=scheduler)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        # if epoch>warmup_epochs:
        scheduler.step()
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X,y, training=False)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                # optimizer.zero_grad()
                # with backpack(DiagHessian(), HMP()):
                # # keep graph for autodiff HVPs
                #     loss.backward()
                # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
                # with open(trace_val_filename,"a") as f:
                #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    total_params = 0
    nonzero_params = 0
    for name, param in model.named_parameters():
        module = dict(model.named_modules()).get(name.rsplit('.', 1)[0], None)
        if module is not None and hasattr(module, 'do_prune') and module.do_prune:
            total_params += param.numel()
            nonzero_params += (param != 0).sum().item()
    sparsity = 1 - nonzero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)


def train_val_loop_ResNet(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, warmup_epochs_2=1, min_epochs=1):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch == warmup_epochs_2:
            print("Changing the learning rate and momentum")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-3
                param_group['momentum'] = 0.98
        if epoch == prune_epoch or (epoch>prune_epoch and (epoch-prune_epoch) % prune_between==0):
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs_2,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        scheduler.step()
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                # optimizer.zero_grad()
                # with backpack(DiagHessian(), HMP()):
                # # keep graph for autodiff HVPs
                #     loss.backward()
                # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
                # with open(trace_val_filename,"a") as f:
                #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)



def train_val_loop_ResNet_scheduler(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, warmup_epochs_2=1, min_epochs=1):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch == prune_epoch+1:
            print("Changing the learning rate and momentum")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-4
                param_group['momentum'] = 0.98
        if epoch == prune_epoch:
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            # elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
            #     print("Pruning the weights")
            #     Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
            #     prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2_ResNet(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs_2,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                # optimizer.zero_grad()
                # with backpack(DiagHessian(), HMP()):
                # # keep graph for autodiff HVPs
                #     loss.backward()
                # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
                # with open(trace_val_filename,"a") as f:
                #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)
        scheduler.step(epoch = epoch, metric = val_acc)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)

def train_val_loop_ResNet_scheduler_v2(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, scheduler_2, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, warmup_epochs_2=1, min_epochs=1):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        # if epoch == warmup_epochs_2:
        #     print("Changing the learning rate and momentum")
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 5e-3
        #         param_group['momentum'] = 0.98
        if epoch == prune_epoch:
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            # elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
            #     print("Pruning the weights")
            #     Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
            #     prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2_ResNet(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs_2,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)
        if epoch > warmup_epochs_2:
            scheduler_2.step(epoch)
        scheduler.step(epoch = epoch, metric = val_acc)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)

def train_val_loop_scheduler(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler_1, scheduler_2, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, min_epochs=1):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            if epoch < warmup_epochs:
                print(f"Epoch: {epoch}| Learning Rate: {scheduler_1.get_last_lr()}", file=f)
            else:
                print(f"Epoch: {epoch}| Learning Rate: {scheduler_2.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch == prune_epoch or (epoch>prune_epoch and (epoch-prune_epoch) % prune_between==0):
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
                print("Pruning the weights")
                Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        if epoch < warmup_epochs:
            scheduler_1.step()
            with open (log_filename,"a") as f:
                print(f"Epoch: {epoch}| Learning Rate: {scheduler_1.get_last_lr()}", file=f)
        else:
            scheduler_2.step(epoch-warmup_epochs)
            with open (log_filename,"a") as f:
                print(f"Epoch: {epoch}| Learning Rate: {scheduler_2.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                # optimizer.zero_grad()
                # with backpack(DiagHessian(), HMP()):
                # # keep graph for autodiff HVPs
                #     loss.backward()
                # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
                # with open(trace_val_filename,"a") as f:
                #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)

def train_val_loop_v2(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, min_epochs=1):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch >= prune_epoch and epoch % prune_between == 0:
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_v2(optimizer, mask=True, mag_prune=mag_prune, filter_based=True, bias_prune=bias_prune)
                prune_count += 1
            elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
                print("Pruning the weights")
                Prune_Score_v2(optimizer, mask=True, mag_prune=mag_prune, filter_based=True, bias_prune=bias_prune)
                prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_v2(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f}", file=f)
        else:    
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        scheduler.step()
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                # optimizer.zero_grad()
                # with backpack(DiagHessian(), HMP()):
                # # keep graph for autodiff HVPs
                #     loss.backward()
                # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
                # with open(trace_val_filename,"a") as f:
                #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)


def train_val_loop_global(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, min_epochs=1):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio:
            no_jenks = True
        if epoch >= prune_epoch and epoch % prune_between == 0:
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if one_shot and epoch==prune_epoch:
                print("Pruning the weights")
                Prune_Score_Global(model,optimizer=optimizer,kill_velocity=kill_velocity,mask=mask,prune_file=prune_filename)
                prune_count += 1
            elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0:
                print("Pruning the weights")
                Prune_Score_Global(model,optimizer=optimizer,kill_velocity=kill_velocity,mask=mask,prune_file=prune_filename)
                prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
        count +=1
        torch.cuda.empty_cache()
        # with prof.profile(use_cuda=True, record_shapes=True) as prof:
        acc, acc5, loss = train_one_step_prune_global(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
        # with open(debug_filename,"a") as f:
        #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
        if mask and epoch>prune_epoch:
                ## Go through all the parameters and set the pruned ones to zero
            for name, param in model.named_parameters():
                param.data = param.data * optimizer.state[param]['mask']
        l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
        with open(train_filename, "a") as f:
            print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        scheduler.step()
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                # optimizer.zero_grad()
                # with backpack(DiagHessian(), HMP()):
                # # keep graph for autodiff HVPs
                #     loss.backward()
                # trace = hutchinson_trace_hmp(model, V=1000, V_batch=10)
                # with open(trace_val_filename,"a") as f:
                #     print(f"Iteration: {count_val}| Trace: {trace: .5f}", file=f)
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)

            # val_loss /= len(test_dataloader)
            # val_acc /= len(test_dataloader)

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}", file=f)


    torch.save(model.state_dict(), f"models/{timestamp}_{experiment_name}_{model_name}_epoch_{epoch}.pth")

    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f) 


def _jenks_gvf_mask(w):
    """Magnitude 2-class Jenks on |w| (prefix-sum SDCM). Returns (gvf, keep_frac, mask) where the
    mask keeps the HIGH-|w| group (the kstar lowest-|w| weights are zeroed). GVF = 1 - SDCM/SDAM."""
    a = w.detach().abs().flatten().float()
    n = a.numel()
    if n < 2:
        return 1.0, 1.0, torch.ones_like(w)
    s, sidx = torch.sort(a)
    s = s.double()
    S1 = torch.zeros(n + 1, dtype=torch.double, device=a.device)
    S2 = torch.zeros(n + 1, dtype=torch.double, device=a.device)
    S1[1:] = torch.cumsum(s, 0); S2[1:] = torch.cumsum(s * s, 0)
    k = torch.arange(1, n, dtype=torch.double, device=a.device)
    left  = S2[1:n] - S1[1:n] ** 2 / k
    right = (S2[n] - S2[1:n]) - (S1[n] - S1[1:n]) ** 2 / (n - k)
    sdcm = left + right
    kstar = int(torch.argmin(sdcm).item()) + 1          # lower group (kstar smallest |w|) -> pruned
    sdam = (S2[n] - S1[n] ** 2 / n).item()
    gvf = float(1 - (sdcm.min().item() / sdam)) if sdam > 0 else 0.0
    keep = (n - kstar) / n
    mask = torch.ones(n, device=w.device, dtype=w.dtype)
    mask[sidx[:kstar]] = 0
    return gvf, keep, mask.reshape(w.shape)


def train_val_loop_GVF(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler,
                       accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp,
                       train_filename, val_filename, log_filename, sparsity_filename, prune_filename,
                       gvf_csv, EPOCHS=40, target_sparsity=0.75, gvf_thresh=0.64, gvf_adaptive=False,
                       use_ema=True, ema_decay=0.95, ema_delay=5, rewind_at_freeze=True,
                       check_from_epoch=0, use_bf16=True, ema_failsafe=True, ema_failsafe_tail=3,
                       gvf_saliency=False):
    """GVF-gated progressive pruning ablation (prune by MAGNITUDE, freeze layer after pruning).
    Every epoch (from check_from_epoch, incl. before the first training step) compute each prunable
    (dim in [2,4]) layer's magnitude-Jenks GVF; any layer with GVF >= gvf_thresh is pruned by
    magnitude and FROZEN (its mask never changes again), optimistically forcing the rest to
    compensate. Once global sparsity >= target_sparsity, pruning stops, LR/WD rewinds, and EMA runs
    the recovery tail. Layerwise (epoch, layer, gvf, keep_frac, pruned_this_epoch, frozen) -> gvf_csv."""
    import csv as _csv
    optimizer.gvf_mode = True                       # external pruning; no per-step Jenks re-masking...
    optimizer.gvf_saliency = gvf_saliency           # ...unless True: not-yet-frozen layers keep the
                                                    # JORTsE dynamic saliency (see optimizer step()).
    prunable = [(n, p) for n, p in model.named_parameters() if p.dim() in (2, 4)]
    for _n, p in model.named_parameters():
        if p not in optimizer.state:
            optimizer.state[p] = {}
        optimizer.state[p]['mask'] = torch.ones_like(p.data, requires_grad=False)
        optimizer.state[p]['frozen'] = False        # set True when GVF-pruned so step() holds it dead
    frozen = set()
    print(f"GVF-gated saliency dynamic on not-yet-frozen layers: {gvf_saliency}")
    with open(gvf_csv, 'w', newline='') as f:
        _csv.writer(f).writerow(['epoch', 'layer', 'gvf', 'keep_frac', 'pruned_this_epoch', 'frozen'])

    def _sparsity():
        nz = sum(torch.count_nonzero(p) for _, p in prunable)
        tot = sum(p.numel() for _, p in prunable)
        return float(1 - nz / tot)

    max_val_acc = 0.0; max_ema_val_acc = 0.0
    ema = None; ema_frozen_epoch = None; target_reached_epoch = None
    sparsity = 0.0
    print(f"GVF-gated: thresh={gvf_thresh} adaptive={gvf_adaptive} target={target_sparsity} "
          f"EMA(decay={ema_decay},delay={ema_delay}) rewind={rewind_at_freeze}")

    failsafe_epoch = EPOCHS - ema_delay - max(ema_failsafe_tail, 1)   # latest freeze that still leaves an EMA tail
    for epoch in range(1, EPOCHS + 1):
        print("Epoch: ", epoch)
        # ---- EMA failsafe: if the GVF criterion never drove sparsity to target, force-freeze the
        # current (partial) masks late enough that EMA still seeds and harvests a recovery tail. ----
        if use_ema and ema_failsafe and target_reached_epoch is None and epoch >= failsafe_epoch:
            target_reached_epoch = epoch
            _sp = _sparsity()
            print(f"[EMA failsafe] target sparsity not reached (at {_sp:.5f}); force-freezing at epoch "
                  f"{epoch}; EMA seeds at {epoch + ema_delay}")
            if rewind_at_freeze and hasattr(scheduler, "reset_accumulators"):
                scheduler.reset_accumulators()
                print(f"Late LR/WD rewind at failsafe freeze (epoch {epoch})")
        # ---- GVF check + prune/freeze (BEFORE training this epoch) ----
        if (epoch - 1) >= check_from_epoch and target_reached_epoch is None:
            gvfs = {}
            for name, p in prunable:
                if p in frozen:
                    continue
                g, keep, m = _jenks_gvf_mask(p.data)
                gvfs[name] = (g, keep, m, p)
            if gvf_adaptive and len(gvfs) >= 4:
                gv = torch.tensor([v[0] for v in gvfs.values()])
                _, _, gm = _jenks_gvf_mask(gv)      # meta-Jenks on the GVF vector; high group -> prune
                nl = list(gvfs.keys())
                to_prune = {nl[i] for i in range(len(nl)) if gm.flatten()[i] > 0}
            else:
                to_prune = {nm for nm, v in gvfs.items() if v[0] >= gvf_thresh}
            for name, p in prunable:
                if name not in gvfs:
                    continue
                g, keep, m, _ = gvfs[name]
                pruned_now = 0
                if name in to_prune and p not in frozen and sparsity < target_sparsity:
                    optimizer.state[p]['mask'] = m.to(p.device)
                    p.data.mul_(optimizer.state[p]['mask'])
                    optimizer.state[p]['frozen'] = True   # step() now holds this layer hard-dead
                    frozen.add(p); pruned_now = 1
                    print(f"  [GVF] prune+freeze {name}: GVF={g:.3f} keep={keep*100:.1f}%")
                with open(gvf_csv, 'a', newline='') as f:
                    _csv.writer(f).writerow([epoch, name, f"{g:.4f}", f"{keep:.4f}", pruned_now, int(p in frozen)])
            sparsity = _sparsity()
            with open(sparsity_filename, "a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
            if sparsity >= target_sparsity and target_reached_epoch is None:
                target_reached_epoch = epoch
                print(f"Target sparsity {sparsity:.5f} reached at epoch {epoch}; EMA seeds at {epoch + ema_delay}")
                if rewind_at_freeze and hasattr(scheduler, "reset_accumulators"):
                    scheduler.reset_accumulators()
                    print(f"Late LR/WD rewind at freeze (epoch {epoch})")

        # ---- train ----
        model.train()
        with open(train_filename, "a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        start = time(); count = 0; tl = ta = t5 = 0.0
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        for X, y in train_dataloader:
            count += 1
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if use_bf16:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = model(X); loss = loss_fn(pred, y)
            else:
                pred = model(X); loss = loss_fn(pred, y)
            loss.backward()
            for p in frozen:                        # keep pruned weights dead (no gradient)
                if p.grad is not None:
                    p.grad.mul_(optimizer.state[p]['mask'])
            optimizer.step()
            for p in frozen:                        # and exactly zero after the step
                p.data.mul_(optimizer.state[p]['mask'])
            ta += accuracy(pred, y).item(); t5 += top5accuracy(pred, y).item(); tl += loss.item()
        with open(train_filename, "a") as f:
            print(f"Iteration: {count}| Loss: {tl/max(count,1): .5f}| Acc: {ta/max(count,1): .5f} | Top 5 Acc: {t5/max(count,1): .5f}", file=f)
        print(f"Time taken for epoch: {time()-start}")
        scheduler.step()
        with open(log_filename, "a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)

        # ---- EMA seed (delay after target) / accumulate over frozen mask ----
        if use_ema and ema_frozen_epoch is None and target_reached_epoch is not None \
           and epoch >= target_reached_epoch + ema_delay:
            ema_frozen_epoch = epoch
            ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
            print(f"EMA seeded at epoch {epoch} from recovered sparse model (target at {target_reached_epoch})")
        if use_ema and ema_frozen_epoch is not None and epoch > ema_frozen_epoch:
            msd = model.state_dict()
            for k in ema:
                v = msd[k]
                if v.dtype.is_floating_point:
                    ema[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                else:
                    ema[k].copy_(v)
            for n_, p_ in model.named_parameters():
                if n_ in ema and p_ in optimizer.state and 'mask' in optimizer.state[p_]:
                    ema[n_].mul_(optimizer.state[p_]['mask'].float())

        # ---- val ----
        model.eval()
        with torch.inference_mode():
            with open(val_filename, "a") as f:
                print(f"Epoch: {epoch}", file=f)
            vl = va = v5 = 0.0; cv = 0
            for X, y in val_dataloader:
                cv += 1; X, y = X.to(device), y.to(device)
                yp = model(X); vl += loss_fn(yp, y).item()
                va += accuracy(yp, y); v5 += top5accuracy(yp, y)
                with open(val_filename, "a") as f:
                    print(f"Iteration: {cv}| Loss: {vl/cv: .5f}| Acc: {va/cv: .5f} | Top 5 Acc {v5/cv}", file=f)
            track_ok = (ema_frozen_epoch is not None and epoch > ema_frozen_epoch) if use_ema else (target_reached_epoch is not None)
            if va/cv > max_val_acc and track_ok:
                max_val_acc = va/cv
                torch.save(model.state_dict(), f"models/best_{timestamp}_{experiment_name}_{model_name}.pth")
            if use_ema and ema_frozen_epoch is not None and epoch > ema_frozen_epoch:
                bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: ema[k].to(bak[k].dtype) for k in ema})
                ea, ec = 0.0, 0
                for Xe, ye in val_dataloader:
                    Xe, ye = Xe.to(device), ye.to(device)
                    ea += (model(Xe).argmax(1) == ye).float().mean(); ec += 1
                ea = ea/ec
                with open(val_filename, "a") as f:
                    print(f"Epoch: {epoch}| EMA Val Acc: {ea: .5f}", file=f)
                if ea > max_ema_val_acc:
                    max_ema_val_acc = ea
                    torch.save(model.state_dict(), f"models/best_ema_{timestamp}_{experiment_name}_{model_name}.pth")
                model.load_state_dict(bak)
        writer.add_scalars("Accuracy", {"train/acc": ta/max(count,1), "val/acc": va/max(cv,1)}, epoch)

    fs = _sparsity()
    # Safety net: always save the final model (so a run that never reaches target_sparsity -> no
    # EMA seed / no plain-best track still leaves a recoverable sparse checkpoint).
    torch.save(model.state_dict(), f"models/final_{timestamp}_{experiment_name}_{model_name}.pth")
    with open(sparsity_filename, "a") as f:
        print(f"Epoch: {epoch}| Sparsity: {fs: .5f}", file=f)
    with open(val_filename, "a") as f:
        print(f"Best validation accuracy achieved: {max_val_acc: .5f}", file=f)
        if use_ema:
            print(f"Best EMA validation accuracy achieved: {max_ema_val_acc: .5f}", file=f)
    print(f"GVF run done. final sparsity {fs:.5f} | best plain {max_val_acc:.5f} | best EMA {max_ema_val_acc:.5f} | target_reached={target_reached_epoch}")


def train_val_loop_HPO(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp,
                   train_filename, val_filename, log_filename, sparsity_filename, prune_filename, debug_filename, jenks_filename,
                   prune_count=0, one_update=False, EPOCHS=100, sparsity=0.0,
                   prune_epoch_list=None, prune_epoch=0, prune_between=1, prune_ratio=0.5, one_shot=False, mask=True,
                   mag_prune=False, bias_prune=False, kill_velocity=False, l2=0.0, lambda_=0.0, warmup_epochs=0, min_epochs=1, elem_bias = False, accum_steps=1, weight_reset=False,
                   use_ema=False, ema_decay=0.95, rewind_at_freeze=False, ema_delay=5):
    no_jenks =False
    l2 = True
    mag_prune = True
    epoch = 0
    names = [name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    name_first = names[0]
    name_last = names[-1]
    imp_names = [name_first, name_last]
    print(f"Prune epoch list: {prune_epoch_list}")
    print(f"Prune epoch: {prune_epoch}")
    print(f"Prune between: {prune_between}")
    max_val_acc = 0.0
    # Tier-1 EMA: epoch-granularity weight averaging (params + BN buffers). Opt-in.
    if use_ema:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        max_ema_val_acc = 0.0
        # EMA must average over a STABLE (frozen) mask. One-shot freezes at prune_epoch;
        # iterative freezes when sparsity first reaches prune_ratio (pruning then stops).
        # Track that epoch and (re)seed the EMA buffer there so no mask-churn epoch enters
        # the running average. Until then EMA does not accumulate or get harvested.
        ema_frozen_epoch = None
        target_reached_epoch = prune_epoch if one_shot else None   # EMA seeds ema_delay epochs AFTER this
        print(f"EMA enabled (decay={ema_decay}, seed delay={ema_delay}); target-reached epoch = {target_reached_epoch}")
    while (sparsity < prune_ratio and epoch<EPOCHS) or epoch<=min_epochs:    # Training loop
        print("Epoch: ", epoch)
        epoch += 1
        model.train()
        #print the epoch and learning rate
        with open(train_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        count = 0
        train_loss, train_acc = 0.0, 0.0
        train_top5acc = 0.0
        start = time()
        print(f"Memory free: {get_memory_free_MiB(0)} MiB")
        if sparsity >= prune_ratio or (one_shot and epoch > prune_epoch):
            no_jenks = True   # Jenks off after the one-shot prune (saliency shaping only pre-prune)
        if epoch == prune_epoch or (epoch>prune_epoch and (epoch-prune_epoch) % prune_between==0):
            # if kill_velocity and epoch==prune_epoch:
            #     Prune_Score(optimizer, kill_velocity=True)
            if not weight_reset:
                if one_shot and epoch==prune_epoch:
                    print("Pruning the weights")
                    Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                    prune_count += 1
                elif not one_shot and epoch>=prune_epoch and sparsity < prune_ratio:   # gradual: cadence controlled by outer (epoch-prune_epoch)%prune_between; removed hardcoded epoch%5==0 that skipped prune_epochs not divisible by 5
                    print("Pruning the weights")
                    Prune_Score_v3(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                    prune_count += 1
            else:
                if one_shot and epoch==prune_epoch:
                    print("Pruning the weights with weight reset")
                    Prune_Score_Reset(model, optimizer, epoch, imp_names, prune_epoch_list, mask=True, 
                                      mag_prune=mag_prune, filter_based=False, bias_prune=bias_prune, prune_file=prune_filename)
                    prune_count += 1
            # if not kill_velocity or not mask:
            #     Prune_Score(optimizer)
            '''Make sure the weights are back on the device'''
            # with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            #     print("Able to prune the weights", file=f)
            # model = prunedmodel.to(device)
            non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
            total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
            sparsity = 1 - non_zero_params / total_params
            with open(sparsity_filename,"a") as f:
                print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
            # Late-EMA freeze detection (iterative path): once the target sparsity is reached
            # pruning stops and the mask is fixed for the rest of training. Seed EMA here.
            if use_ema and (not one_shot) and target_reached_epoch is None and sparsity >= prune_ratio:
                target_reached_epoch = epoch
                print(f"Target sparsity {float(sparsity): .5f} reached at epoch {epoch}; "
                      f"EMA will seed at epoch {epoch + ema_delay} (after sparse net + BN stats recover)")
                # Late warm-restart tied to the (dynamic) freeze epoch: by now the auto-Jenks LR
                # has decayed far down, leaving the frozen sparse net no headroom to recover. Reset
                # the LR/WD accumulators to init so the schedule re-anneals over the frozen-mask tail.
                if rewind_at_freeze and hasattr(scheduler, "reset_accumulators"):
                    scheduler.reset_accumulators()
                    print(f"Late LR/WD rewind at freeze (epoch {epoch}): scheduler accumulators reset to init")
        if one_update:
            count +=1
            torch.cuda.empty_cache()
            # with prof.profile(use_cuda=True, record_shapes=True) as prof:
            acc, acc5, loss = train_one_step_prune_HPO(model,train_dataloader, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks, bias_prune=bias_prune, filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename, mag=False, elem_bias=elem_bias, accumulation_steps=accum_steps)
            # with open(debug_filename,"a") as f:
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), file=f)
            if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                for name, param in model.named_parameters():
                    param.data = param.data * optimizer.state[param]['mask']
            l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
            lr_prune = sum(torch.norm(p)**2 for p in model.parameters() if p.dim() in [2, 4])
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {loss: .5f}| Acc: {acc.item(): .5f} | Top 5 Acc: {acc5.item(): .5f} |L_2: {l2_reg: .5f} | L_R: {lr_prune: .5f}", file=f)
            # one_update path returns per-epoch train metrics; record so summary/TB are not 0
            train_loss, train_acc, train_top5acc = float(loss), acc.item(), acc5.item()
        else:
            for X, y in train_dataloader:
                # print(torch.cuda.memory_summary())
                torch.cuda.empty_cache()
                count += 1
                # loss = loss.clone() + lambda_ * l2_reg
                X, y = X.to(device), y.to(device)
                master_count += 1
                acc, acc5, loss = train_one_step_prune(model,X, y, optimizer, loss_fn, epoch, warmup_epochs,prune_epochs=prune_epoch,no_jenks=no_jenks ,filter_based=False, mask=mask, L2 = l2, lambda_=lambda_, debug = True, debugfile = debug_filename, jenksfile=jenks_filename)
                if mask and epoch>prune_epoch:
                    ## Go through all the parameters and set the pruned ones to zero
                    for name, param in model.named_parameters():
                        param.data = param.data * optimizer.state[param]['mask']
                # acc = accuracy(y_pred, y)
                # acc_5 = top5accuracy(y_pred, y)
                train_loss += loss.item()
                train_top5acc += acc5.item()
                train_acc += acc.item()
                l2_reg = sum(torch.norm(p) ** 2 for p in model.parameters())
                # print("Train loss type : ", type(train_loss))
                # print("Train Acc type : ", type(train_acc))
                # print("Train Top5Acc type : ", type(train_top5acc))
                # print("Loss type : ", type(loss))
                # print("l2_reg type : ", type(l2_reg))
            with open(train_filename, "a") as f:
                print(f"Iteration: {count}| Loss: {train_loss/count: .5f}| Acc: {train_acc/count: .5f} | Top 5 Acc: {train_top5acc/count: .5f} |L_2: {l2_reg: .5f}", file=f)
        stop = time()
        print(f"Time taken for epoch: {stop-start}")
        # if epoch < 151:
        scheduler.step()
        with open (log_filename,"a") as f:
            print(f"Epoch: {epoch}| Learning Rate: {scheduler.get_last_lr()}", file=f)
        # Seed EMA once, ema_delay epochs AFTER target sparsity, from the RECOVERED sparse model so
        # weights AND BatchNorm running stats are mutually consistent. Seeding on the raw abrupt prune
        # (OVER_PRUNE=0 -> one-shot jump to the Jenks break) pinned EMA at ~random: collapsed seed +
        # EMA'd BN stats lagging the sparse weights.
        if use_ema and ema_frozen_epoch is None and target_reached_epoch is not None \
           and epoch >= target_reached_epoch + ema_delay:
            ema_frozen_epoch = epoch
            ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
            print(f"EMA seeded at epoch {epoch} from recovered sparse model (target reached at {target_reached_epoch})")
        if use_ema and ema_frozen_epoch is not None and epoch > ema_frozen_epoch:
            _msd = model.state_dict()
            for _k in ema:
                _v = _msd[_k]
                if _v.dtype.is_floating_point:
                    ema[_k].mul_(ema_decay).add_(_v.detach().float(), alpha=1.0-ema_decay)
                else:
                    ema[_k].copy_(_v)
            if mask:
                for _n, _pp in model.named_parameters():
                    if _n in ema and _pp in optimizer.state and 'mask' in optimizer.state[_pp]:
                        ema[_n].mul_(optimizer.state[_pp]['mask'].float())
        # if epoch == warmup_epochs:
        #     '''Change the learning rate to the base value'''
        #     for group in optimizer.param_groups:
        #         group['lr'] = 3e-3
            # for param_group in optimizer.param_groups:
            #     param_group['momentum'] = 0.99
            
        model.eval()
        with torch.inference_mode():
            with open(val_filename,"a") as f:
                print(f"Epoch: {epoch}", file=f)
            val_loss, val_acc = 0.0, 0.0
            val_top5acc = 0.0
            count_val = 0
            for X, y in val_dataloader:
                count_val += 1
                X, y = X.to(device), y.to(device)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                acc = accuracy(y_pred, y)
                top5_acc = top5accuracy(y_pred, y)
                val_top5acc += top5_acc
                val_acc += acc
                with open(val_filename,"a") as f:
                    print(f"Iteration: {count_val}| Loss: {val_loss/count_val: .5f}| Acc: {val_acc/count_val: .5f} | Top 5 Acc {val_top5acc/count_val}", file=f)
            # Track the plain best only once the mask is at its final (frozen) sparsity, so the
            # reported/saved best is a genuine final-sparsity checkpoint and NOT a mid-churn,
            # lower-sparsity peak. Non-EMA callers keep the legacy epoch>prune_epoch behavior.
            _track_ok = (ema_frozen_epoch is not None and epoch > ema_frozen_epoch) if use_ema else (epoch > prune_epoch)
            if val_acc/count_val > max_val_acc and _track_ok:
                max_val_acc = val_acc/count_val
                torch.save(model.state_dict(), f"models/best_{timestamp}_{experiment_name}_{model_name}.pth")
            # EMA is evaluated on its OWN schedule (every epoch after the freeze), independent of
            # whether plain set a new best. The frozen-mask tail is exactly where EMA's variance
            # reduction pays off, and plain may never beat its pre-freeze (lower-sparsity) peak.
            if use_ema and ema_frozen_epoch is not None and epoch>ema_frozen_epoch:
                _bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict({k: ema[k].to(_bak[k].dtype) for k in ema})
                _eacc, _ec = 0.0, 0
                for Xe, ye in val_dataloader:
                    Xe, ye = Xe.to(device), ye.to(device)
                    _eacc += (model(Xe).argmax(1) == ye).float().mean(); _ec += 1
                _eacc = _eacc/_ec
                with open(val_filename,"a") as f:
                    print(f"Epoch: {epoch}| EMA Val Acc: {_eacc: .5f}", file=f)
                if _eacc > max_ema_val_acc:
                    max_ema_val_acc = _eacc
                    torch.save(model.state_dict(), f"models/best_ema_{timestamp}_{experiment_name}_{model_name}.pth")
                model.load_state_dict(_bak)
        _ntr = max(count, 1); _nva = max(count_val, 1)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss/_ntr, "val/loss": val_loss/_nva}, global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc/_ntr, "val/acc": val_acc/_nva}, global_step=epoch)
        with open("LeNet300_100_MNIST_output/output_(1).txt","a") as f:
            print(f"Epoch: {epoch}| Train loss: {train_loss/_ntr: .5f}| Train acc: {train_acc/_ntr: .5f}| Val loss: {val_loss/_nva: .5f}| Val acc: {val_acc/_nva: .5f}", file=f)



    val_loss, val_acc = 0.0, 0.0
    val_top5acc = 0.0
    count_val = 0
    '''Make sure the weights are back on the device'''
    non_zero_params = sum(torch.count_nonzero(p) for p in model.parameters() if p.dim() in [2, 4])
    total_params = sum(p.numel() for p in model.parameters() if p.dim() in [2, 4])
    sparsity = 1 - non_zero_params / total_params
    with open(sparsity_filename,"a") as f:
        print(f"Epoch: {epoch}| Sparsity: {sparsity: .5f}", file=f)
    with open(val_filename,"a") as f:
        print(f"Best validation accuracy achieved: {max_val_acc: .5f}", file=f)
        if use_ema:
            print(f"Best EMA validation accuracy achieved: {max_ema_val_acc: .5f}", file=f)