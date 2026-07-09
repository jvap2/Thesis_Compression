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


def train_val_loop_HPO(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, accuracy, top5accuracy, writer, device, experiment_name, model_name, timestamp, 
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
                elif not one_shot and epoch>=prune_epoch and epoch % 5 == 0 and sparsity < prune_ratio:
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