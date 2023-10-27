# -*- coding: utf-8 -*-

import re
import os
import csv


def main():
    # 1、parameter configuration
    quality_list = [1, 2, 3, 4]
    dataset_name = "openimages"  # openimages or coco
    result_csv_save_path = "./collect_coding_time_{}.csv".format(dataset_name)
    encode_time_regx = re.compile(
        "(.*)enc_time:([0-9.]+)")
    decode_time_regx = re.compile(
        "(.*)dec_time:([0-9.]+)")

    # 2、collect data
    with open(result_csv_save_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['quality', "encode_time", "decode_time"])
        for quality in quality_list:
            encoder_log_dir = "../../logs/default/Encoder/{}/{}".format(dataset_name, quality)
            decoder_log_dir = "../../logs/default/Decoder/{}/{}".format(dataset_name, quality)
            all_encode_time_list = []
            all_decode_time_list = []
            # (1) collect encode time
            encode_log_list = os.listdir(encoder_log_dir)
            for encode_log in encode_log_list:
                encode_log_path = os.path.join(encoder_log_dir, encode_log)
                with open(encode_log_path) as f:
                    lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    if (match := encode_time_regx.match(line)):
                        _, encode_time = match.groups()
                        # print(encode_time)
                        all_encode_time_list.append(float(encode_time))

            # (2) collect decode time
            decode_log_list = os.listdir(decoder_log_dir)
            for decode_log in decode_log_list:
                decode_log_path = os.path.join(decoder_log_dir, decode_log)
                with open(decode_log_path) as f:
                    lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    if (match := decode_time_regx.match(line)):
                        _, decode_time = match.groups()
                        # print(decode_time)
                        all_decode_time_list.append(float(decode_time))

        # 3、deal data
        print(len(all_encode_time_list))
        print(len(all_decode_time_list))
        # encode time
        all_encode_time = sum(all_encode_time_list)
        # decode time
        all_decode_time = sum(all_decode_time_list)
        writer.writerow([quality, format(all_encode_time, '.10f'), format(all_decode_time, '.10f')])


if __name__ == '__main__':
    main()
