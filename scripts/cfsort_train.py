import time

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from Multi_channel_AE_celfeer import *
from sklearn.model_selection import KFold
import sys, os
import numpy as np
from datetime import datetime
import logging
import gc
import tensorflow as tf
import collections

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

num_nodes = [1024, 512, 32]
rate_dropout = [0.05, 0.01, 0.0]
learning_rate = 0.001
batch_size = 32
loss = "mae"
optimizer = "Adam"
epoch = 100
global_seed = 1

# 设置随机种子
tf.random.set_seed(global_seed)

# 假设num_features和TOTAL_TRAINING_SAMPLES是已知的


# 设置模型参数
model_parameter = {
    "num_nodes": num_nodes,
    "rate_dropout": rate_dropout,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "loss": loss,
    "optimizer": optimizer,
    "epoch": epoch,
    "epoch_start": 0,
    "existing_model": "None",
    "new_model": True,
    "initializer": "default",
}


def test_cfsort_function(train_x, train_y, test_x, test_y, output_directory, file, input_path, simi, mode):
    l1_scores = []
    ccc_scores = []
    preds = []
    num_features = train_x.size(1)
    feature_shape = (num_features,)
    for i in range(train_x.shape[2]):  # 针对每个通道单独训练
        train_channel_x = train_x[:, :, i]
        test_channel_x = test_x[:, :, i]

        avg_training_loss, avg_validation_loss, pred = NN_multiple_response_batch_variedLR_startMiddle(
            model_parameter, train_channel_x, train_y, test_channel_x, test_y, output_directory, feature_shape, file,
            input_path, simi, mode, i
        )
        pred = torch.tensor(pred.numpy(), dtype=torch.float64)
        l1, ccc = score(pred.cpu().detach().numpy(), test_y.cpu().detach().numpy())
        l1_scores.append(l1)
        ccc_scores.append(ccc)
        preds.append(pred)

    # 将结果拼接起来
    final_l1 = np.mean(l1_scores)
    final_ccc = np.mean(ccc_scores)
    final_pred = np.mean(preds, axis=0)

    return final_l1, final_ccc, final_pred


def NN_multiple_response_batch_variedLR_startMiddle(model_parameter, train_x, train_y, test_x, test_y, output_prefix,
                                                    feature_shape, file, input_path, simi, mode, channel):
    fixed_learning_rate = False,
    test_x = tf.convert_to_tensor(test_x, dtype=tf.float64)
    n_classes = train_y.shape[1] if len(train_y.shape) > 1 else 1
    if model_parameter["new_model"]:
        model = tf.keras.Sequential()
        n_layers = len(model_parameter["num_nodes"])
        initializer = model_parameter["initializer"]

        for i_layer in range(n_layers):
            model.add(tf.keras.layers.Dense(model_parameter["num_nodes"][i_layer]))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation(tf.nn.relu))
            model.add(tf.keras.layers.Dropout(model_parameter["rate_dropout"][i_layer]))

        model.add(tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax))
    else:
        EXISTING_MODEL = model_parameter["existing_model"]
        model = tf.keras.models.load_model(EXISTING_MODEL)
        print("LOAD EXISTING MODEL:", EXISTING_MODEL)
    batch_size = model_parameter["batch_size"]
    steps_per_epoch = len(train_x) // batch_size
    TOTAL_TRAINING_SAMPLES = train_x.shape[0]
    model_step = round(float(TOTAL_TRAINING_SAMPLES) / batch_size) - 2
    # 确保 model_parameter["step"] 合理
    if model_step > steps_per_epoch:
        print("警告: model_parameter['step'] 大于数据集的步数。这可能导致 OutOfRangeError。")
        model_step = steps_per_epoch

    if fixed_learning_rate:
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_parameter["learning_rate"])
    else:
        optimizer = tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=model_parameter["learning_rate"],
                decay_steps=model_step,
                decay_rate=0.95
            )
        )

    if not model.built:
        model.compile(optimizer=optimizer, loss=model_parameter["loss"], metrics=['accuracy'])  # 根据需要添加其他指标

    loss_train_list = []
    loss_test_list = []
    patience = 10
    wait = 0
    best = 100000
    EPOCH_START = model_parameter["epoch_start"]
    avg_training_loss = 0
    avg_validation_loss = 0
    y_pred = 0


    for epoch in range(EPOCH_START, model_parameter["epoch"]):
        # print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        tmp_loss_train_list = []
        tmp_loss_test_list = []

        # 生成数据集
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(500000).batch(batch_size)

        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            if step >= model_step:
                break

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                # if channel == 0:
                #     total_params = model.count_params()
                #     print(f"模型的总参数数量: {total_params}")
                logits = tf.cast(logits, tf.float64)
                loss = compute_loss(logits, y_batch_train, model_parameter["loss"])

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            y_pred = model.predict(test_x, verbose=0)
            y_pred = tf.cast(y_pred, tf.float64)
            loss_test = compute_loss(y_pred, test_y, model_parameter["loss"])
            tmp_loss_train_list.append(float(loss))
            tmp_loss_test_list.append(float(loss_test))
            # if step % 200 == 0:
            #     print("Training loss (for one batch) at step %d: %.8f" % (step, float(loss)))
            #     print("Validation loss (for one batch) at step %d: %.8f" % (step, float(loss_test)))

        avg_training_loss = float(np.mean(tmp_loss_train_list))
        avg_validation_loss = float(np.mean(tmp_loss_test_list))

        # print("Average training loss over epoch: %.8f" % (avg_training_loss))
        # print("Average validation loss over epoch: %.8f" % (avg_validation_loss))

        loss_train_list.append(avg_training_loss)
        loss_test_list.append(avg_validation_loss)

        gc.collect()
        wait += 1

        if avg_validation_loss < best:
            best = avg_validation_loss
            model.save(output_prefix + "/cfsort" + input_path + str(simi) + mode + file + str(channel) + ".model.h5")
            wait = 0
        if wait >= patience:
            print("wait enough", epoch)
            break

    output_after_training_results(model, loss_test_list, loss_train_list, y_pred, output_prefix)

    # 返回最终的评估结果
    return avg_training_loss, avg_validation_loss, y_pred


def compute_loss(logits, labels, loss_type):
    if loss_type == "mae":
        labels = labels.numpy()
        return tf.reduce_mean(tf.abs(logits - labels))
    elif loss_type == "mse":
        return tf.reduce_mean(tf.square(logits - labels))
    else:
        raise ValueError("Unsupported loss type")


def output_after_training_results(model, loss_test_list, loss_train_list, y_pred, output_prefix):
    np.savetxt(output_prefix + ".final.y_pred.txt", y_pred, delimiter="\t", fmt="%f")
    np.savetxt(output_prefix + ".loss_train.txt", loss_train_list, delimiter="\t", fmt="%f")
    np.savetxt(output_prefix + ".loss_test.txt", loss_test_list, delimiter="\t", fmt="%f")


def extract_values(file_name):
    # Split the filename based on the boolean value (True/False)
    if "True" in file_name:
        sparse, rest = file_name.split("True")
        ifrare = "True"
    else:
        sparse, rest = file_name.split("False")
        ifrare = "False"
    rare = rest
    return float(sparse), ifrare, float(rare)


def main():
    # 直接在代码中指定参数
    # input_path = "UXM_39.bed"  # WGBS_ref_2+1.txt
    output_directory = "./results"
    num_samples = 8  # 52
    unknowns = 0
    parallel_job_id = 1
    simi = 1
    train = 1
    test = 1  # WGBS不用这个这个就可以直接设置为0 这个就是没有proportions
    for input_path in ["./data/ALS.txt"]:  # "WGBS_ref_1+2.txt", "WGBS_ref_2+1.txt" UXM_39.bed
        for file in ["0.3False0.4"]:
            #  "0.1False0.4", "0.3False0.4","0.5False0.4", "0.3True0.1", "0.3True0.2", "0.3True0.3"
            sparse, ifrare, rare = extract_values(file)
            proportions = "./results/fractions/sample_array_100019_spar" + file + ".pkl"  #
            # 这里
            mode = "overall"  # "high-resolution"  # high-resolution(dictionary  overall
            np.random.seed(parallel_job_id)
            num_tissues = 19
            if not os.path.exists(output_directory) and parallel_job_id == 1:
                os.makedirs(output_directory)
                print("made " + output_directory + "/")
                print()
            else:
                print("writing to " + output_directory + "/")

            # data_df = pd.read_csv(input_path, header=None, delimiter="\t")

            data_df = pd.read_csv(input_path, delimiter="\t", header=None,
                                  skiprows=1)  # read input samples/reference data

            # 删除指定的列（263, 264, 265）
            # columns_to_drop = [263, 264, 265]
            # data_df = data_df.drop(columns=columns_to_drop)

            print(f"finished reading {input_path}", "file", file)
            print()

            print(f"beginning generation of {output_directory}")
            print()

            if train == 1:
                props = pkl.load(open(proportions, 'rb'))
                if simi == 2:
                    # simulated 2 data
                    simulated_x_1, true_beta_1, simulated_x_2, true_beta_2 = simulate_arrays(data_df,
                                                                                             int(num_samples),
                                                                                             int(unknowns),
                                                                                             proportions=props,
                                                                                             simi=simi,
                                                                                             sparse=sparse,
                                                                                             ifrare=ifrare,
                                                                                             rare=rare)
                    data_1 = torch.tensor(simulated_x_1)
                    label_1 = torch.tensor(props[:int(props.shape[0] / 2), :])
                    data_2 = torch.tensor(simulated_x_2)
                    label_2 = torch.tensor(props[int(props.shape[0] / 2):, :])
                    train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(data_1, label_1, test_size=0.1,
                                                                                random_state=0)
                    train_x_2, test_x_2, train_y_2, test_y_2 = train_test_split(data_2, label_2, test_size=0.1,
                                                                                random_state=0)

                    train_x = torch.cat((train_x_1, train_x_2), dim=0)
                    train_y = torch.cat((train_y_1, train_y_2), dim=0)

                    # # Generate random permutation of indices
                    # random_indices = torch.randperm(train_x.size(0))

                    # # Use indexing to reorder the tensor
                    # train_x = train_x[random_indices]
                    # train_y = train_y[random_indices]

                    test_x = torch.cat((test_x_1, test_x_2), dim=0)
                    test_y = torch.cat((test_y_1, test_y_2), dim=0)
                    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
                else:
                    x, true_beta = simulate_arrays(data_df, int(num_samples), int(unknowns), proportions=props,
                                                   simi=simi, sparse=sparse, ifrare=ifrare, rare=rare)
                    data = torch.tensor(x)
                    label = torch.tensor(props)
                    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.1, random_state=0)
                L1 = []
                CCC = []
                start_time = time.time()
                l1, ccc, pred = test_cfsort_function(train_x, train_y, test_x, test_y,
                                                     output_directory=output_directory, file=file,
                                                     input_path=input_path, simi=simi, mode=mode)
                end_time = time.time()
                print('l1, ccc:', l1, ccc, "time", end_time-start_time)
                L1.append(l1)
                CCC.append(ccc)
                with open(output_directory + "/cfsort" + input_path + str(simi) + mode + file + "alpha_est.pkl",
                          "wb") as f:
                    pkl.dump(pred, f)
                with open(
                        output_directory + "/cfsort" + input_path + str(simi) + mode + file + "alpha_true.pkl",
                        "wb") as f:
                    pkl.dump(props, f)

                if simi == 2:
                    with open(
                            output_directory + "/cfsort" + input_path + str(simi) + mode + file + "beta_true_1.pkl",
                            "wb") as f:
                        pkl.dump(true_beta_1, f)
                    with open(
                            output_directory + "/cfsort" + input_path + str(simi) + mode + file + "beta_true_2.pkl",
                            "wb") as f:
                        pkl.dump(true_beta_2, f)
                else:
                    with open(output_directory + "/cfsort" + input_path + str(simi) + mode + file + "beta_true.pkl",
                              "wb") as f:
                        pkl.dump(true_beta, f)

            if test == 1:
                x, true_beta, test_samples, training_props = simulate_arrays(data_df, int(num_samples),
                                                                             int(unknowns), sparse=sparse,
                                                                             ifrare=ifrare, rare=rare)
                preds = []
                for i in range(test_samples.shape[2]):  # 针对每个通道单独训练
                    test_channel_x = test_samples[:, :, i]
                    model_path = output_directory + "/cfsort" + input_path + str(simi) + mode + file + str(
                        i) + ".model.h5"
                    model = tf.keras.models.load_model(model_path)
                    # 进行预测
                    pred = model.predict(test_channel_x, verbose=0)
                    preds.append(pred)

                # 将结果拼接起来
                final_pred = np.mean(preds, axis=0)
                # 保存预测结果
                with open(output_directory + "/testcfsort" + input_path + str(simi) + mode + file + "alpha_est.pkl",
                          "wb") as f:
                    pkl.dump(final_pred, f)


if __name__ == "__main__":
    main()
