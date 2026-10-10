import csv
import logging
import matplotlib
from scipy.io import savemat

matplotlib.use('TkAgg')  # 或者可以选择其他交互式后端，如 'Qt5Agg'、'TkAgg' 等
import matplotlib.pyplot as plt
from model.SAM import *
import torch.nn.functional as nn
from dataloader import *
# from medpy import metric
import cv2
from torch.utils.data.dataset import ConcatDataset
from utils.fastMRI_supportingFunctions import *
# %%


def get_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    # if logger.handlers:
    #    return logger
    # create file handler which logs even debug messages
    fh = logging.FileHandler(path + '/' + name + '.txt')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s:%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# %%
def model_selection(model_name, args):
    from my_model import network
    if model_name == 'backbone':
        args.train_ps = 320
        args.embed_dim = 32
        args.win_size = 4
        args.token_projection = 'linear'
        args.token_mlp = 'leff'

        model = network(args)

    return model

# %%

def trainer(args, model, save_path):
    logger = get_logger("training_log", save_path)
    logger.info('trainer')
    logger.info(str(args))
    num_params = print_network(model)
    logger.info("Total number of parameters: " + str(num_params))

    dataset_dir1 = r'/mnt/f/lz/datasets/AioDATA/Brain/T1flair'
    train_data_name1 = 'train'
    train_maps_name1 = 'train_cms'
    val_data_name1 = 'val'
    val_maps_name1 = 'val_cms'

    dataset_dir2 = r'/mnt/f/lz/datasets/AioDATA/Brain/T1'
    train_data_name2 = 'train'
    train_maps_name2 = 'train_cms'
    val_data_name2 = 'val'
    val_maps_name2 = 'val_cms'

    dataset_dir3 = r'/mnt/f/lz/datasets/AioDATA/Brain/T2'
    train_data_name3 = 'train'
    train_maps_name3 = 'train_cms'
    val_data_name3 = 'val'
    val_maps_name3 = 'val_cms'

    dataset_dir4 = r'/mnt/f/lz/datasets/AioDATA/CMR/CMR_T1'
    train_data_name4 = 'train'
    train_maps_name4 = 'train_maps'
    val_data_name4 = 'val'
    val_maps_name4 = 'val_maps'

    dataset_dir5 = r'/mnt/f/lz/datasets/AioDATA/CMR/CMR_T2'
    train_data_name5 = 'train'
    train_maps_name5 = 'train_maps'
    val_data_name5 = 'val'
    val_maps_name5 = 'val_maps'

    dataset_dir6 = r'/mnt/f/lz/datasets/AioDATA/Knee/fastmri-knee/knee_PD'
    train_data_name6 = 'train'
    train_maps_name6 = 'train_maps'
    val_data_name6 = 'val'
    val_maps_name6 = 'val_maps'

    dataset_dir7 = r'/mnt/f/lz/datasets/AioDATA/Knee/fastmri-knee/knee_PDFS'
    train_data_name7 = 'train'
    train_maps_name7 = 'train_maps'
    val_data_name7 = 'val'
    val_maps_name7 = 'val_maps'



    data_list1 = load_train_data(dataset_dir1, train_data_name1, train_maps_name1, val_data_name1, val_maps_name1)
    data_list1['train'] = data_list1['train']
    data_list1['val'] = data_list1['val']
    Dataset_train1 = MRIDataset_b(data_list1['train'], acceleration=4, center_fraction=args.center_fraction)
    Dataset_val1 = MRIDataset_b(data_list1['val'], acceleration=4, center_fraction=args.center_fraction)


    data_list2 = load_train_data(dataset_dir2, train_data_name2, train_maps_name2, val_data_name2, val_maps_name2)
    data_list2['train'] = data_list2['train']
    data_list2['val'] = data_list2['val']
    Dataset_train2 = MRIDataset_b(data_list2['train'], acceleration=4, center_fraction=args.center_fraction)
    Dataset_val2 = MRIDataset_b(data_list2['val'], acceleration=4, center_fraction=args.center_fraction)


    data_list3 = load_train_data(dataset_dir3, train_data_name3, train_maps_name3, val_data_name3, val_maps_name3)
    data_list3['train'] = data_list3['train']
    data_list3['val'] = data_list3['val']
    Dataset_train3 = MRIDataset_b(data_list3['train'], acceleration=4, center_fraction=args.center_fraction)
    Dataset_val3 = MRIDataset_b(data_list3['val'], acceleration=4, center_fraction=args.center_fraction)


    data_list4 = load_train_data(dataset_dir4, train_data_name4, train_maps_name4, val_data_name4, val_maps_name4)
    data_list4['train'] = data_list4['train']
    data_list4['val'] = data_list4['val']
    Dataset_train4 = MRIDataset_c(data_list4['train'], acceleration=8, center_fraction=args.center_fraction)
    Dataset_val4 = MRIDataset_c(data_list4['val'], acceleration=8, center_fraction=args.center_fraction)


    data_list5 = load_train_data(dataset_dir5, train_data_name5, train_maps_name5, val_data_name5, val_maps_name5)
    data_list5['train'] = data_list5['train']
    data_list5['val'] = data_list5['val']
    Dataset_train5 = MRIDataset_c(data_list5['train'], acceleration=8, center_fraction=args.center_fraction)
    Dataset_val5 = MRIDataset_c(data_list5['val'], acceleration=8, center_fraction=args.center_fraction)


    data_list6 = load_train_data(dataset_dir6, train_data_name6, train_maps_name6, val_data_name6, val_maps_name6)
    data_list6['train'] = data_list6['train']
    data_list6['val'] = data_list6['val']
    Dataset_train6 = MRIDataset_k(data_list6['train'], acceleration=6, center_fraction=args.center_fraction)
    Dataset_val6 = MRIDataset_k(data_list6['val'], acceleration=6, center_fraction=args.center_fraction)


    data_list7 = load_train_data(dataset_dir7, train_data_name7, train_maps_name7, val_data_name7, val_maps_name7)
    data_list7['train'] = data_list7['train']
    data_list7['val'] = data_list7['val']
    Dataset_train7 = MRIDataset_k(data_list7['train'], acceleration=6, center_fraction=args.center_fraction)
    Dataset_val7 = MRIDataset_k(data_list7['val'], acceleration=6, center_fraction=args.center_fraction)


    concat_dataset = ConcatDataset([Dataset_train1, Dataset_train2, Dataset_train3, Dataset_train4, Dataset_train5, Dataset_train6, Dataset_train7])
    trainloader = DataLoader(concat_dataset, shuffle=True, batch_size=args.batch_size_train, num_workers=args.num_workers)
    logger.info("training samples " + str(len(trainloader)))


    val_dataset_1 = ConcatDataset([Dataset_val1, Dataset_val2, Dataset_val3])
    val_dataset_2 = ConcatDataset([Dataset_val4, Dataset_val5])
    val_dataset_3 = ConcatDataset([Dataset_val6, Dataset_val7])


    best_psnr = 0.0
    train_psnr_aver = []
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_vali_epoch = 0

    if args.optimizer == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=args.lr_decay)

    model_sam = IntroduceSAM().cuda()
    model_sam.eval()

    epoch_start = 0
    if args.resume_training != 0:
        load_model_path = os.path.join(save_path, 'current.pth')
        data = torch.load(load_model_path)
        model.load_state_dict(data["model_state_dict"])
        optimizer.load_state_dict(data["optimizer_state_dict"])
        scheduler.load_state_dict(data['scheduler'])
        epoch_start = data["epoch"] + 1

        logger.info("resume training, epoch_start=" + str(epoch_start))


    for epoch_num in range(epoch_start, max_epoch):
        iter_num = 0
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info('epoch %d : lr: %f' % (epoch_num, current_lr))
        model.train()

        for iteration, samples in enumerate(trainloader):
            iter_num = iter_num + 1
            img_und, img_gt, rawdata_und, masks, sensitivity, filename = samples
            img_gt = torch.tensor(img_gt).cuda()
            img_und = torch.tensor(img_und).cuda()
            rawdata_und = torch.tensor(rawdata_und).cuda()
            masks = torch.tensor(masks).cuda()
            sensitivity = torch.tensor(sensitivity).cuda()

            zf = T.complex_abs(img_und).data.to('cpu')
            sense_und = T.complex_abs(img_und).data.to('cpu').numpy()
            error_gt = img_gt.permute(0, 3, 1, 2) - img_und.permute(0, 3, 1, 2)

            with torch.no_grad():
                sam_map = model_sam(zf.unsqueeze(0))

            outputs, em = model(img_und, rawdata_und, masks, sensitivity, sam_map)
            loss = nn.mse_loss(outputs, img_gt, size_average=False, reduce=True) + 0.1*nn.mse_loss(em, error_gt, size_average=False, reduce=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sense_recon = T.complex_abs(outputs).data.to('cpu').numpy()
            sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()

            base_psnr = psnr(abs(sense_gt[0]), abs(sense_und[0]))
            base_ssim = ssim(abs(sense_gt[0]), abs(sense_und[0]))
            base_nmse = nmse(abs(sense_gt[0]), abs(sense_und[0]))
            train_psnr = psnr(abs(sense_gt[0]), abs(sense_recon[0]))
            train_ssim = ssim(abs(sense_gt[0]), abs(sense_recon[0]))
            train_nmse = nmse(abs(sense_gt[0]), abs(sense_recon[0]))
            print('iteration {} out of {} in train, base_psnr={:.4f}, train_psnr={:.4f}, base_ssim={:.4f}, train_ssim={:.4f}, base_nmse={:.4f}, train_nmse={:.4f}'.format(iter_num, epoch_num, base_psnr,
                                                                                           train_psnr, base_ssim, train_ssim, base_nmse, train_nmse))
            train_psnr_aver.append(train_psnr)

        # Validation
        model.eval()

        base_all_in_one_psnr = []
        base_all_in_one_ssim = []
        base_all_in_one_nmse = []

        all_in_one_psnr = []
        all_in_one_ssim = []
        all_in_one_nmse = []

        val_datasets = [val_dataset_1, val_dataset_2, val_dataset_3]

        for task_i in range(3):
            if task_i == 0:
                print("start test brain:")
            elif task_i == 1:
                print("start test cmr:")
            elif task_i == 2:
                print("start test knee:")


            Dataset_val = val_datasets[task_i]
            valloader = DataLoader(Dataset_val, shuffle=True, batch_size=args.batch_size_eval, num_workers=args.num_workers)

            count = 0
            val_psnr_aver = []
            base_psnr_aver = []
            val_ssim_aver = []
            base_ssim_aver = []
            val_nmse_aver = []
            base_nmse_aver = []

            for iteration, samples in enumerate(valloader):
                count = count + 1
                img_und, img_gt, rawdata_und, masks, sensitivity, filename = samples
                img_gt = torch.tensor(img_gt).cuda()
                img_und = torch.tensor(img_und).cuda()
                rawdata_und = torch.tensor(rawdata_und).cuda()
                masks = torch.tensor(masks).cuda()
                sensitivity = torch.tensor(sensitivity).cuda()

                zf = T.complex_abs(img_und).data.to('cpu')
                sense_und = T.complex_abs(img_und).data.to('cpu').numpy()

                with torch.no_grad():
                    sam_map_val = model_sam(zf.unsqueeze(0))
                    rec, erromap = model(img_und, rawdata_und, masks, sensitivity, sam_map_val)

                sense_recon = T.complex_abs(rec).data.to('cpu').numpy()
                sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()

                base_psnr2 = psnr(abs(sense_gt[0]), abs(sense_und[0]))
                base_ssim2 = ssim(abs(sense_gt[0]), abs(sense_und[0]))
                base_nmse2 = nmse(abs(sense_gt[0]), abs(sense_und[0]))
                val_psnr2 = psnr(abs(sense_gt[0]), abs(sense_recon[0]))
                val_ssim2 = ssim(abs(sense_gt[0]), abs(sense_recon[0]))
                val_nmse2 = nmse(abs(sense_gt[0]), abs(sense_recon[0]))

                print(
                    'task{},  iteration {} out of {} in val, base_psnr={:.4f}, train_psnr={:.4f}, base_ssim={:.4f}, train_ssim={:.4f}, base_nmse={:.4f}, train_nmse={:.4f}'.format(
                        task_i, count, epoch_num, base_psnr2,
                        val_psnr2, base_ssim2, val_ssim2, base_nmse2, val_nmse2))

                base_psnr_aver.append(base_psnr2)
                val_psnr_aver.append(val_psnr2)
                base_ssim_aver.append(base_ssim2)
                val_ssim_aver.append(val_ssim2)
                base_nmse_aver.append(base_nmse2)
                val_nmse_aver.append(val_nmse2)

            ave_val_psnr = np.mean(val_psnr_aver)
            ave_base_psnr = np.mean(base_psnr_aver)
            ave_val_ssim = np.mean(val_ssim_aver)
            ave_base_ssim = np.mean(base_ssim_aver)
            ave_val_nmse = np.mean(val_nmse_aver)
            ave_base_nmse = np.mean(base_nmse_aver)

            logger.info(
                "task:{}  current_epoch:{} base_val_psnr:{:.4f} val_psnr:{:.4f} base_val_ssim:{:.4f} val_ssim:{:.4f} base_val_nmse:{:.4f} val_nmse:{:.4f}".format(
                    task_i, epoch_num, ave_base_psnr,
                    ave_val_psnr, ave_base_ssim, ave_val_ssim, ave_base_nmse, ave_val_nmse))

            base_all_in_one_psnr.append(ave_base_psnr)
            base_all_in_one_ssim.append(ave_base_ssim)
            base_all_in_one_nmse.append(ave_base_nmse)

            all_in_one_psnr.append(ave_val_psnr)
            all_in_one_ssim.append(ave_val_ssim)
            all_in_one_nmse.append(ave_val_nmse)

        ave_base_all_in_one_psnr = np.mean(base_all_in_one_psnr)
        ave_base_all_in_one_ssim = np.mean(base_all_in_one_ssim)
        ave_base_all_in_one_nmse = np.mean(base_all_in_one_nmse)
        ave_all_in_one_psnr = np.mean(all_in_one_psnr)
        ave_all_in_one_ssim = np.mean(all_in_one_ssim)
        ave_all_in_one_nmse = np.mean(all_in_one_nmse)

        logger.info(
            "current_epoch:{} ave_base_val_psnr:{:.4f} ave_val_psnr:{:.4f} ave_base_val_ssim:{:.4f} ave_val_ssim:{:.4f} ave_base_val_nmse:{:.4f} ave_val_nmse:{:.4f} ave_best_psnr:{:.4f}".format(
                epoch_num, ave_base_all_in_one_psnr,
                ave_all_in_one_psnr, ave_base_all_in_one_ssim, ave_all_in_one_ssim, ave_base_all_in_one_nmse, ave_all_in_one_nmse,
                best_psnr))


        scheduler.step()

        if ave_all_in_one_psnr > best_psnr:
            # save the best model
            best_psnr = ave_all_in_one_psnr
            best_vali_epoch = epoch_num
            save_model_path = os.path.join(save_path, 'best.pth')
            torch.save(model.state_dict(), save_model_path)



        if (epoch_num + 1) % 10 == 0:  # 输出结果
            save_model_path = os.path.join(save_path, f'{epoch_num + 1}.pth')
            torch.save(model.state_dict(), save_model_path)

        # —————————————————————————— save current model and checkpoint————————————————————
        data_save = {"args": args,
                     "epoch": epoch_num,
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "best_psnr": best_psnr
                     }

        save_model_path = os.path.join(save_path, 'current.pth')
        torch.save(data_save, save_model_path)
        logger.info("{} save current model to {}".format("epoch " + str(epoch_num), save_model_path))
    # done
    logger.info("best_val_epoch " + str(best_vali_epoch))



def evaluation(args, model, save_path):
    logger = get_logger("test_log", save_path)
    logger.info('evaluation')
    logger.info(str(args))

    dataset_dir1 = r'/data01/code/mx/All-in-one/AioDATA/Brain/T1'
    test_data_name1 = 'test'
    test_maps_name1 = 'test_cms'

    dataset_dir2 = r'/data01/code/mx/All-in-one/AioDATA/Brain/T2'
    test_data_name2 = 'test'
    test_maps_name2 = 'test_cms'

    dataset_dir3 = r'/data01/code/mx/All-in-one/AioDATA/Brain/T1flair'
    test_data_name3 = 'test'
    test_maps_name3 = 'test_cms'

    dataset_dir4 = r'/data01/code/mx/All-in-one/AioDATA/CMR/CMR_T1'
    test_data_name4 = 'test'
    test_maps_name4 = 'test_maps'

    dataset_dir5 = r'/data01/code/mx/All-in-one/AioDATA/CMR/CMR_T2'
    test_data_name5 = 'test'
    test_maps_name5 = 'test_maps'

    dataset_dir6 = r'/data01/code/mx/All-in-one/AioDATA/Knee/fastmri-knee/knee_PD'
    test_data_name6 = 'test'
    test_maps_name6 = 'test_maps'

    dataset_dir7 = r'/data01/code/mx/All-in-one/AioDATA/Knee/fastmri-knee/knee_PDFS'
    test_data_name7 = 'test'
    test_maps_name7 = 'test_maps'

    dataset_dir8 = r'/data01/code/mx/All-in-one/AioDATA/Knee/fastmri-knee/knee_add'  # add testset
    test_data_name8 = 'test'
    test_maps_name8 = 'test_maps'


    data_list1 = load_testdata_path(dataset_dir1, test_data_name1, test_maps_name1)
    data_list1['test'] = data_list1['test']
    Dataset_test1 = MRIDataset_b(data_list1['test'], acceleration=4, center_fraction=args.center_fraction)


    data_list2 = load_testdata_path(dataset_dir2, test_data_name2, test_maps_name2)
    data_list2['test'] = data_list2['test']
    Dataset_test2 = MRIDataset_b(data_list2['test'], acceleration=4, center_fraction=args.center_fraction)


    data_list3 = load_testdata_path(dataset_dir3, test_data_name3, test_maps_name3)
    data_list3['test'] = data_list3['test']
    Dataset_test3 = MRIDataset_b(data_list3['test'], acceleration=4, center_fraction=args.center_fraction)


    data_list4 = load_testdata_path(dataset_dir4, test_data_name4, test_maps_name4)
    data_list4['test'] = data_list4['test']
    Dataset_test4 = MRIDataset_c(data_list4['test'], acceleration=8, center_fraction=args.center_fraction)


    data_list5 = load_testdata_path(dataset_dir5, test_data_name5, test_maps_name5)
    data_list5['test'] = data_list5['test']
    Dataset_test5 = MRIDataset_c(data_list5['test'], acceleration=8, center_fraction=args.center_fraction)


    data_list6 = load_testdata_path(dataset_dir6, test_data_name6, test_maps_name6)
    data_list6['test'] = data_list6['test']
    Dataset_test6 = MRIDataset_k(data_list6['test'], acceleration=6, center_fraction=args.center_fraction)


    data_list7 = load_testdata_path(dataset_dir7, test_data_name7, test_maps_name7)
    data_list7['test'] = data_list7['test']
    Dataset_test7 = MRIDataset_k(data_list7['test'], acceleration=6, center_fraction=args.center_fraction)


    data_list8 = load_testdata_path(dataset_dir8, test_data_name8, test_maps_name8)
    data_list8['test'] = data_list8['test']
    Dataset_test8 = MRIDataset_k(data_list8['test'], acceleration=6, center_fraction=args.center_fraction)


    all_base_psnr, all_base_ssim, all_base_nmse = [], [], []   # add
    all_test_psnr, all_test_ssim, all_test_nmse = [], [], []

    model_sam = IntroduceSAM().cuda()
    model_sam.eval()

    model.load_state_dict(torch.load(os.path.join(save_path, 'best.pth')))
    model.eval()

    test_dataset_1 = ConcatDataset([Dataset_test1, Dataset_test2, Dataset_test3])
    test_dataset_2 = ConcatDataset([Dataset_test4, Dataset_test5])
    test_dataset_3 = ConcatDataset([Dataset_test6, Dataset_test7, Dataset_test8])

    test_datasets = [test_dataset_1, test_dataset_2, test_dataset_3]

    for task_i in range(3):
        if task_i == 0:
            print("start test brain:")
        elif task_i == 1:
            print("start test cmr:")
        elif task_i == 2:
            print("start test knee:")


        Dataset_test = test_datasets[task_i]
        testloader = DataLoader(Dataset_test, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

        count = 0
        test_psnr_aver = []
        base_psnr_aver = []
        test_ssim_aver = []
        base_ssim_aver = []
        test_nmse_aver = []
        base_nmse_aver = []

        for iteration, samples in enumerate(testloader):
            count = count + 1
            img_und, img_gt, rawdata_und, masks, sensitivity, rawdata_name = samples

            img_gt = torch.tensor(img_gt).cuda()
            img_und = torch.tensor(img_und).cuda()
            rawdata_und = torch.tensor(rawdata_und).cuda()
            masks = torch.tensor(masks).cuda()
            sensitivity = torch.tensor(sensitivity).cuda()
            gt = T.complex_abs(img_gt).data.to('cpu')
            zf = T.complex_abs(img_und).data.to('cpu')

            with torch.no_grad():
                sammap = model_sam(zf.unsqueeze(0))
                rec, erromap = model(img_und, rawdata_und, masks, sensitivity, sammap)
               


            sense_recon = T.complex_abs(rec).data.to('cpu').numpy()
            sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()
            sense_und = T.complex_abs(img_und).data.to('cpu').numpy()

            base_psnr = psnr(abs(sense_gt[0]), abs(sense_und[0]))
            base_ssim = ssim(abs(sense_gt[0]), abs(sense_und[0]))
            base_nmse = nmse(abs(sense_gt[0]), abs(sense_und[0]))
            test_psnr = psnr(abs(sense_gt[0]), abs(sense_recon[0]))
            test_ssim = ssim(abs(sense_gt[0]), abs(sense_recon[0]))
            test_nmse = nmse(abs(sense_gt[0]), abs(sense_recon[0]))
            # ================= Task 内收集 =================
            base_psnr_aver.append(base_psnr)
            base_ssim_aver.append(base_ssim)
            base_nmse_aver.append(base_nmse)
            test_psnr_aver.append(test_psnr)
            test_ssim_aver.append(test_ssim)
            test_nmse_aver.append(test_nmse)

            # ================= 全局收集 ================= add
            all_base_psnr.append(base_psnr)
            all_base_ssim.append(base_ssim)
            all_base_nmse.append(base_nmse)
            all_test_psnr.append(test_psnr)
            all_test_ssim.append(test_ssim)
            all_test_nmse.append(test_nmse)

            # ----------------- 构造唯一文件名 -----------------
            rawdata_name = rawdata_name[0]
            filename = os.path.basename(rawdata_name)  # e.g., id01_1.mat
            parent = os.path.basename(os.path.dirname(rawdata_name))  # test
            parent2 = os.path.basename(os.path.dirname(os.path.dirname(rawdata_name)))  # T1
            parent3 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(rawdata_name))))  # Brain
            save_name = f"{parent3}_{parent2}_{filename.replace('.mat', '')}"
            print("DEBUG:", rawdata_name, parent3, parent2, parent, filename)

            # 定义保存目录
            rec_dir = '/data01/code/mx/All-in-one/CAPNet/result468/yiyuan_xianzhu_analyse/rec'
            gt_dir = '/data01/code/mx/All-in-one/CAPNet/result468/yiyuan_xianzhu_analyse/gt'
            mat_dir = '/data01/code/mx/All-in-one/CAPNet/result468/yiyuan_xianzhu_analyse/recmat'

            # 如果目录不存在则自动创建
            os.makedirs(rec_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(mat_dir, exist_ok=True)

            # csv_file = os.path.join(save_path, f"{args.net_name}_results.csv")
            csv_file = os.path.join('/data01/code/mx/All-in-one/CAPNet/result468/yiyuan_xianzhu_analyse', f"capnet_results.csv")
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([save_name, task_i, count, test_psnr, test_ssim, test_nmse])

            # ----------------- 保存重建图像 -----------------
            rec_play = abs(sense_recon[0]) * 255
            cv2.imwrite(os.path.join(rec_dir, f"{save_name}_rec.png"), rec_play)

            # ----------------- 保存 GT 图像 -----------------
            gt_play = abs(sense_gt[0]) * 255
            cv2.imwrite(os.path.join(gt_dir, f"{save_name}_gt.png"), gt_play)

            # ----------------- 保存 .mat 文件 -----------------
            savemat(
                os.path.join(mat_dir,
                             f"{save_name}=psnr={test_psnr:.4f}_ssim={test_ssim:.4f}_nmse={test_nmse:.4f}.mat"),
                {"recover_image": abs(sense_recon[0])})

            print('task {}, id {}, in test, base_psnr={}, test_psnr={}, base_rmse={}，test_rmse={}, base_ssim={}, test_ssim={}'.format(
                task_i, count, base_psnr, test_psnr, base_nmse, test_nmse, base_ssim, test_ssim))

            logger.info('task {}, id {}, in test, base_psnr={}, test_psnr={}, base_rmse={}，test_rmse={}, base_ssim={}, test_ssim={}'.format(
                task_i, count, base_psnr, test_psnr, base_nmse, test_nmse, base_ssim, test_ssim))

        # ----------------- 每个 task 打印均值 ± std -----------------
        ave_test_psnr, std_test_psnr = np.mean(test_psnr_aver), np.std(test_psnr_aver)
        ave_base_psnr, std_base_psnr = np.mean(base_psnr_aver), np.std(base_psnr_aver)
        ave_test_ssim, std_test_ssim = np.mean(test_ssim_aver), np.std(test_ssim_aver)
        ave_base_ssim, std_base_ssim = np.mean(base_ssim_aver), np.std(base_ssim_aver)
        ave_test_nmse, std_test_nmse = np.mean(test_nmse_aver), np.std(test_nmse_aver)
        ave_base_nmse, std_base_nmse = np.mean(base_nmse_aver), np.std(base_nmse_aver)

        print(
            f"----------task:{task_i}\n"
            f"PSNR (base): {ave_base_psnr:.4f} ± {std_base_psnr:.4f}, "
            f"PSNR (ours): {ave_test_psnr:.4f} ± {std_test_psnr:.4f}\n"
            f"SSIM (base): {ave_base_ssim:.4f} ± {std_base_ssim:.4f}, "
            f"SSIM (ours): {ave_test_ssim:.4f} ± {std_test_ssim:.4f}\n"
            f"NMSE (base): {ave_base_nmse:.4f} ± {std_base_nmse:.4f}, "
            f"NMSE (ours): {ave_test_nmse:.4f} ± {std_test_nmse:.4f}"
        )

        logger.info(
            f"----------task:{task_i}\n"
            f"PSNR (base): {ave_base_psnr:.4f} ± {std_base_psnr:.4f}, "
            f"PSNR (ours): {ave_test_psnr:.4f} ± {std_test_psnr:.4f}\n"
            f"SSIM (base): {ave_base_ssim:.4f} ± {std_base_ssim:.4f}, "
            f"SSIM (ours): {ave_test_ssim:.4f} ± {std_test_ssim:.4f}\n"
            f"NMSE (base): {ave_base_nmse:.4f} ± {std_base_nmse:.4f}, "
            f"NMSE (ours): {ave_test_nmse:.4f} ± {std_test_nmse:.4f}"
        )

    # ================= 全局统计 =================
    ave_base_all_in_one_psnr, std_base_all_in_one_psnr = np.mean(all_base_psnr), np.std(all_base_psnr)
    ave_base_all_in_one_ssim, std_base_all_in_one_ssim = np.mean(all_base_ssim), np.std(all_base_ssim)
    ave_base_all_in_one_nmse, std_base_all_in_one_nmse = np.mean(all_base_nmse), np.std(all_base_nmse)

    ave_all_in_one_psnr, std_all_in_one_psnr = np.mean(all_test_psnr), np.std(all_test_psnr)
    ave_all_in_one_ssim, std_all_in_one_ssim = np.mean(all_test_ssim), np.std(all_test_ssim)
    ave_all_in_one_nmse, std_all_in_one_nmse = np.mean(all_test_nmse), np.std(all_test_nmse)

    print(
        ">>>>>In test: Overall across all tasks:\n"
        f"PSNR (base): {ave_base_all_in_one_psnr:.4f} ± {std_base_all_in_one_psnr:.4f}, "
        f"PSNR (ours): {ave_all_in_one_psnr:.4f} ± {std_all_in_one_psnr:.4f}\n"
        f"SSIM (base): {ave_base_all_in_one_ssim:.4f} ± {std_base_all_in_one_ssim:.4f}, "
        f"SSIM (ours): {ave_all_in_one_ssim:.4f} ± {std_all_in_one_ssim:.4f}\n"
        f"NMSE (base): {ave_base_all_in_one_nmse:.4f} ± {std_base_all_in_one_nmse:.4f}, "
        f"NMSE (ours): {ave_all_in_one_nmse:.4f} ± {std_all_in_one_nmse:.4f}"
    )

    logger.info(
        ">>>>>In test: Overall across all tasks:\n"
        f"PSNR (base): {ave_base_all_in_one_psnr:.4f} ± {std_base_all_in_one_psnr:.4f}, "
        f"PSNR (ours): {ave_all_in_one_psnr:.4f} ± {std_all_in_one_psnr:.4f}\n"
        f"SSIM (base): {ave_base_all_in_one_ssim:.4f} ± {std_base_all_in_one_ssim:.4f}, "
        f"SSIM (ours): {ave_all_in_one_ssim:.4f} ± {std_all_in_one_ssim:.4f}\n"
        f"NMSE (base): {ave_base_all_in_one_nmse:.4f} ± {std_base_all_in_one_nmse:.4f}, "
        f"NMSE (ours): {ave_all_in_one_nmse:.4f} ± {std_all_in_one_nmse:.4f}"
    )
    # print(
    #     ">>>>>In test: ave_base_val_psnr:{:.4f} ave_val_psnr:{:.4f} ave_base_val_ssim:{:.4f} ave_val_ssim:{:.4f} ave_base_val_nmse:{:.4f} ave_val_nmse:{:.4f}".format(
    #         ave_base_all_in_one_psnr,
    #         ave_all_in_one_psnr, ave_base_all_in_one_ssim, ave_all_in_one_ssim, ave_base_all_in_one_nmse,
    #         ave_all_in_one_nmse))
    # logger.info(
    #     ">>>>>In test: ave_base_val_psnr:{:.4f} ave_val_psnr:{:.4f} ave_base_val_ssim:{:.4f} ave_val_ssim:{:.4f} ave_base_val_nmse:{:.4f} ave_val_nmse:{:.4f}".format(
    #         ave_base_all_in_one_psnr,
    #         ave_all_in_one_psnr, ave_base_all_in_one_ssim, ave_all_in_one_ssim, ave_base_all_in_one_nmse,
    #         ave_all_in_one_nmse))
    
    
    # ==========================================
# 前列腺专属小样本微调流程
# ==========================================
def trainer_finetune(args, model, save_path):
    logger = get_logger("finetune_log", save_path)
    logger.info('--- Start Few-Shot Fine-tuning on Prostate ---')
    
    # 1. 加载满级大佬的预训练权重 (核心！)
    pretrained_path = os.path.join("result468", "backbone", "best.pth")
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        logger.info("✅ 成功加载多器官联合预训练权重！准备开始降维打击！")
    else:
        logger.info("❌ 找不到预训练权重，请检查 result468 路径！")
        return

    # 2. 锁定前列腺数据集 (只喂这 20 张图)
    dataset_dir_p = r'/data01/code/mx/All-in-one/AioDATA/fastmri_prostate_h5_files'
    data_list_p = load_train_data(dataset_dir_p, 'train', 'train_cms', 'val', 'val_cms')
    
    # 假设前列腺我们测试 4倍 加速
    Dataset_train_p = MRIDataset_p(data_list_p['train'], acceleration=4, center_fraction=args.center_fraction)
    Dataset_val_p = MRIDataset_p(data_list_p['val'], acceleration=4, center_fraction=args.center_fraction)

    trainloader = DataLoader(Dataset_train_p, shuffle=True, batch_size=args.batch_size_train, num_workers=args.num_workers)
    valloader = DataLoader(Dataset_val_p, shuffle=False, batch_size=args.batch_size_eval, num_workers=args.num_workers)

    # 3. 设置极小的学习率 (微调关键！原来是 2e-4，现在降低 20 倍)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    model_sam = IntroduceSAM().cuda()
    model_sam.eval()
    
    best_psnr = 0.0
    max_epoch = 30 # 图少，跑30轮足够收敛了
    
    for epoch_num in range(max_epoch):
        model.train()
        train_psnr_aver = []
        
        for iteration, samples in enumerate(trainloader):
            img_und, img_gt, rawdata_und, masks, sensitivity, filename = samples
            img_gt = torch.tensor(img_gt).cuda()
            img_und = torch.tensor(img_und).cuda()
            rawdata_und = torch.tensor(rawdata_und).cuda()
            masks = torch.tensor(masks).cuda()
            sensitivity = torch.tensor(sensitivity).cuda()

            zf = T.complex_abs(img_und).data.to('cpu')
            error_gt = img_gt.permute(0, 3, 1, 2) - img_und.permute(0, 3, 1, 2)

            with torch.no_grad():
                sam_map = model_sam(zf.unsqueeze(0)) 

            outputs, em = model(img_und, rawdata_und, masks, sensitivity, sam_map)
            
            loss = nn.mse_loss(outputs, img_gt, size_average=False, reduce=True) + 0.1*nn.mse_loss(em, error_gt, size_average=False, reduce=True)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算平时成绩
            sense_recon = T.complex_abs(outputs).data.to('cpu').numpy()
            sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()
            train_psnr = psnr(abs(sense_gt[0]), abs(sense_recon[0]))
            train_psnr_aver.append(train_psnr)
            print(f"Epoch {epoch_num}, Iter {iteration+1}/20: Train PSNR = {train_psnr:.4f}")
            
        # ================= 期中考试 (验证集) =================
        model.eval()
        val_psnr_aver = []
        for iteration, samples in enumerate(valloader):
            img_und, img_gt, rawdata_und, masks, sensitivity, filename = samples
            img_gt = torch.tensor(img_gt).cuda()
            img_und = torch.tensor(img_und).cuda()
            rawdata_und = torch.tensor(rawdata_und).cuda()
            masks = torch.tensor(masks).cuda()
            sensitivity = torch.tensor(sensitivity).cuda()
            
            zf = T.complex_abs(img_und).data.to('cpu')
            with torch.no_grad():
                sam_map_val = model_sam(zf.unsqueeze(0))
                rec, erromap = model(img_und, rawdata_und, masks, sensitivity, sam_map_val)
                
            sense_recon = T.complex_abs(rec).data.to('cpu').numpy()
            sense_gt = T.complex_abs(img_gt).data.to('cpu').numpy()
            val_psnr2 = psnr(abs(sense_gt[0]), abs(sense_recon[0]))
            val_psnr_aver.append(val_psnr2)
            
        ave_val_psnr = np.mean(val_psnr_aver)
        logger.info(f"🏆 Epoch {epoch_num} 结束: Val PSNR 平均分 = {ave_val_psnr:.4f}")
        
        # 优胜劣汰保存
        if ave_val_psnr > best_psnr:
            best_psnr = ave_val_psnr
            save_model_path = os.path.join(save_path, 'best_finetuned.pth')
            torch.save(model.state_dict(), save_model_path)
            logger.info(f"🌟 破纪录啦！保存当前最强微调权重！")







