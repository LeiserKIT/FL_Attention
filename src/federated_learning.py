# ----------
# imports
# ----------

import fl_utils
import torch
import torch.nn as nn
import random
import datetime
from tensorboardX import SummaryWriter


def main(model, sim, num_samples, rounds, epochs_per_round, bs, gc):
    # ----------
    # params
    # ----------

    start_time = datetime.datetime.now()

    MODEL = model
    NUM_SAMPLES = num_samples        # None for full dataset
    EPOCHS_PER_ROUND = epochs_per_round
    BATCH_SIZE = bs
    SCALE_PERCENT = 10
    FED_ROUNDS = rounds
    FRACTION = 1.0
    GRADCAM = gc
    SIM = sim

    assert MODEL in ['DenseNet121', 'ResNet101'], 'Invalid model. Choose between DenseNet121 and ResNet101.'
    if NUM_SAMPLES is not None:
        log_num_samples = NUM_SAMPLES
    else:
        log_num_samples = 'all'

    OUTPUT_DIR = 'logs/FED_{}_SIM{}_RD{}_{}_E{}_BS{}_SC{}_'.format(MODEL[0], SIM, FED_ROUNDS, log_num_samples, EPOCHS_PER_ROUND, BATCH_SIZE, SCALE_PERCENT) + datetime.datetime.now().strftime("%d.%m.%y-%H:%M:%S") 
    print(OUTPUT_DIR)
    writer = SummaryWriter(logdir=OUTPUT_DIR)
    writer.add_text('Model', MODEL)
    writer.add_text('Batch Size', str(BATCH_SIZE))
    writer.add_text('Epochs', str(EPOCHS_PER_ROUND))
    writer.add_text('Scaling Percentage', str(SCALE_PERCENT))
    writer.add_text('Num Samples', str(NUM_SAMPLES))


    # ----------
    # load datasets
    # ----------

    train_datasets, test_dataset, trainloaders, validloaders, testloader = fl_utils.load_data(sim=SIM, sc=SCALE_PERCENT, num_samples=NUM_SAMPLES, bs=BATCH_SIZE)


    # ----------
    # load model
    # ----------

    if MODEL == 'DenseNet121':
        model = fl_utils.DenseNet121()
    elif MODEL == 'ResNet101':
        model = fl_utils.ResNet101()


    # ----------
    # set up GPU
    # ----------

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1]
    model = nn.DataParallel(model, device_ids=device_ids).to(DEVICE)


    # ----------
    # federated training
    # ----------

    for i in range(FED_ROUNDS):
        print('ROUND {} START'.format(i+1))

        params = [None] * len(trainloaders)

        # sample random fraction of clients
        client_selection = sorted(random.sample(range(len(trainloaders)), round(len(trainloaders) * FRACTION)))
        print('Number of clients: {}'.format(len(client_selection)))
        print('')

        # send current model weights to clients and train locally
        for j in client_selection:
            print('Round {}/{}. Client {}: Start Training Round'.format(i+1, FED_ROUNDS, j))
            params[j], loss_avg_val, accuracy_val = fl_utils.train(model, trainloaders[j], validloaders[j], epochs=EPOCHS_PER_ROUND, device=DEVICE, cid=j, k=i+1, fed_rounds=FED_ROUNDS)
            writer.add_scalar('val/fed_loss_{}'.format(j), loss_avg_val, i+1)
            writer.add_scalar('val/fed_acc_{}'.format(j), accuracy_val, i+1)
            print('')
            print('Round {}/{}. Client {}: End Training Round'.format(i+1, FED_ROUNDS, j))
            print('')

        # return updates to server (federated averaging)
        first_idx = [idx for idx in range(len(params)) if params[idx] != None][0]
        last_idx = [idx for idx in range(len(params)) if params[idx] != None][-1]
        for key in params[first_idx]:
            weights, weightn = [], []
            for k in client_selection:
                weights.append(params[k][key]*train_datasets[k].__len__())
                weightn.append(train_datasets[k].__len__())
            # weighted averaging model weights
            params[last_idx][key] = sum(weights) / sum(weightn) 

        # initialize updated model for next round
        if MODEL == 'DenseNet121':
            model = fl_utils.DenseNet121()
        elif MODEL == 'ResNet101':
            model = fl_utils.ResNet101()
        model = nn.DataParallel(model, device_ids=device_ids).to(DEVICE)
        model.load_state_dict(params[last_idx])

        print('ROUND {} END'.format(i+1))
        print('')

    print('Done Training')
    print('')


    # ----------
    # global model testing
    # ----------

    if GRADCAM:
        fl_utils.test(model, testloader, test_dataset, writer, output_dir=OUTPUT_DIR, model_str=MODEL, gradcam=GRADCAM, device=DEVICE)


    # time
    end_time = datetime.datetime.now()
    delta = end_time - start_time
    s = delta.seconds
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_elapsed = '{:02}:{:02}:{:02}h'.format(int(hours), int(minutes), int(seconds))
    writer.add_text('Runtime', time_elapsed) 

    ### END OF RUN
    writer.close()
    print('\ndone in {}'.format(time_elapsed))



# mod: 'DenseNet121', 'ResNet101'
# num_samples = None for all data
if __name__ == "__main__":
    main(model='ResNet101', sim=3, num_samples=None, rounds=10, epochs_per_round=4, bs=16, gc=True)