from config import cfg
from model import get_mobilenet_1
import torch
from dataloader import get_dataloader
import tqdm
import statistics
import os
from torcheval.metrics import R2Score

'''
this function is made to manage all the training pipeline, including validation. It also save models checkpoint after each validation.
args:
    None
return:
    None
'''
def train():
    torch.manual_seed(0)

    # loading model, loss and optimizer
    model = get_mobilenet_1()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    metric_speed = R2Score().to("cuda:0")
    metric_steer = R2Score().to("cuda:0")

    # create dataloader
    train_dataloader, validation_dataloader = get_dataloader()

    # create list to deal with all results and save them
    train_list_loss = []
    train_list_acc = []
    train_list_acc2 = []

    val_list_loss = []
    val_list_acc = []
    val_list_acc2 = []

    loss_history = []
    acc_history = []
    acc2_history = []

    # index of checkpoint and bath counting creation
    checkpoint_count = 0
    batch_after_last_validation = 0
    for epoch in range(cfg.TRAIN.NBR_EPOCH):
        
        # creation of index to count gradian accumulation since the last weights update
        gradiant_accumulation_count = 0

        # loop en batchs
        train_range = tqdm.tqdm(train_dataloader)
        for i, (datas_rgb, datas_g, labels_speed, labels_steer) in enumerate(train_range):

            # change model mode
            model.train()
            gradiant_accumulation_count += 1
            batch_after_last_validation += 1

            # moove variables place in memory on GPU RAM
            if torch.cuda.is_available():
                datas_rgb = datas_rgb.cuda()
                datas_g = datas_g.cuda()
                labels_speed = labels_speed.cuda()
                labels_steer = labels_steer.cuda()

            # make prediction
            outputs_1, outputs_2 = model(datas_rgb, datas_g)

            # compute loss
            loss = criterion(outputs_1, labels_speed) + criterion(outputs_2, labels_steer)

            # make gradiant retropropagation
            loss.backward()

            # condition to choose if you update model's weights or not
            if gradiant_accumulation_count >= cfg.TRAIN.GRADIANT_ACCUMULATION or i == len(train_dataloader) - 1:

                # reinitialisation of gradiant accumulation index
                gradiant_accumulation_count = 0

                # update model's weights
                optimizer.step()
                optimizer.zero_grad()

                metric_speed.update(torch.squeeze(outputs_1, dim=-1), labels_speed.cuda())
                metric_steer.update(torch.squeeze(outputs_2, dim=-1), labels_steer)

                # save loss and metric for the current batch
                train_list_loss.append(loss.item())
                train_list_acc.append(metric_speed.compute().item())
                train_list_acc2.append(metric_steer.compute().item())

                metric_speed.reset()
                metric_steer.reset()

                # update tqdm line information
                train_range.set_description("TRAIN -> epoch: %4d || loss: %4.4f || R2_speed: %4.4f || R2_steer: %4.4f" % (epoch, statistics.mean(train_list_loss), statistics.mean(train_list_acc) , statistics.mean(train_list_acc2)))
                train_range.refresh()

            # condition to choose if you have to do a validation or not
            if batch_after_last_validation + 1 > len(train_dataloader)/cfg.TRAIN.VALIDATION_RATIO:

                # remove gradiants computation for the validation
                with torch.no_grad():
                    batch_after_last_validation = 0

                    # validation loop
                    for j, (val_datas_rgb, val_datas_g, val_labels_speed, val_labels_steer) in enumerate(validation_dataloader):
                        
                        # change model mode
                        model.eval()

                        # moove variables place in memory on GPU RAM
                        if torch.cuda.is_available():
                            val_datas_rgb = val_datas_rgb.cuda()
                            val_datas_g = val_datas_g.cuda()
                            val_labels_speed = val_labels_speed.cuda()
                            val_labels_steer = val_labels_steer.cuda()
                        
                        # make prediction
                        outputs_1, outputs_2 = model(val_datas_rgb, val_datas_g)

                        # compute loss
                        loss = criterion(outputs_1, val_labels_speed) + criterion(outputs_2, val_labels_steer)

                        # compute mectric
                        metric_speed.update(torch.squeeze(outputs_1, dim=-1), val_labels_speed)
                        metric_steer.update(torch.squeeze(outputs_2, dim=-1), val_labels_steer)

                        # save loss and metric for the current batch
                        val_list_acc.append(metric_speed.compute().item())
                        val_list_acc2.append(metric_steer.compute().item())

                        metric_speed.reset()
                        metric_steer.reset()

                        # save loss and metric for the current batch
                        val_list_loss.append(loss.item())

                    # print validation results
                    print(" ")
                    print("VALIDATION -> epoch: %4d || loss: %4.4f || R2_speed: %4.4f || R2_steer: %4.4f" % (epoch, statistics.mean(val_list_loss), statistics.mean(val_list_acc) , statistics.mean(val_list_acc2)))

                    # save model checkpoint
                    torch.save(model.state_dict(), os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH,'ckpt_' + str(checkpoint_count)) + ".ckpt")
                    checkpoint_count += 1
                    print(" ")

                    # save loss and metric for the current epoch
                    loss_history.append(statistics.mean(val_list_loss))
                    acc_history.append(statistics.mean(val_list_acc))
                    acc2_history.append(statistics.mean(val_list_acc2))

                    with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'result_history.txt'), 'a+') as result_file:
                        result_file.write(f"checkpoint_{checkpoint_count} : loss = {statistics.mean(val_list_loss)} , R2_speed = {statistics.mean(val_list_acc)} , R2_speed = {statistics.mean(val_list_acc2)} \n")

                    # print loss and metric history
                    print("loss history:")
                    print(loss_history)
                    print("R2 speed history:")
                    print(acc_history)
                    print("R2 steer history:")
                    print(acc2_history)

                    # clear storage of short term result
                    train_list_loss = []
                    train_list_acc = []
                    train_list_acc2 = []
                    val_list_loss = []
                    val_list_acc = []
                    val_list_acc2 = []
        
        # clear storage of short term result
        train_list_loss = []
        train_list_acc = []
        train_list_acc2 = []