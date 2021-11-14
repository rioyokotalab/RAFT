import os
import random
import time

# D = 8
# I = 3
# B = 6
# RI = min(I, B)
# FITRI = (RI + 1) / 2 + (R1 + 1) % 2
# RMNB = min(B - RI, (D - ((FITRI - (RI + 1) % 2) * 2) + 1) / 2)
# FRMNITRB = (RMNB + 1) / 2
# RB = RI + RMNB
# FRNMITR = D - (FITRI + FITRB) * 2
# FITR = FITRI + FITRB + FRNMITR


def calc_iter_nums(data_size, bufsize, initial):
    print(f"D: {data_size} I: {initial} B: {bufsize}")
    # 実際の初期値
    real_initial = min(bufsize, initial)
    # bufのサイズがreal_initialの値になるまでに必要なイテレーション数
    for_iter_num_real_initial = (real_initial + 1) // 2
    # 初めの値が出力されるまでのイテレーション数
    for_iter_num_print_initial = for_iter_num_real_initial + (real_initial + 1) % 2
    is_end_iter = for_iter_num_print_initial > (data_size // 2)
    for_iter_num_print_initial = min(for_iter_num_print_initial, data_size // 2)
    # 残りのbufのサイズ
    remain_bufsize = min(bufsize - real_initial,
                         (data_size - for_iter_num_real_initial * 2) // 2)
    # 残りのbufのサイズを埋めるために必要なイテレーション数
    for_iter_num_remain_buf = remain_bufsize
    # 実際のbufのサイズ
    real_bufsize = real_initial + remain_bufsize
    # bufに対して操作しないイテレーション数
    for_remain_iter_num = data_size - (for_iter_num_remain_buf +
                                       for_iter_num_real_initial) * 2
    for_remain_iter_num = max(0, for_remain_iter_num)
    # 全てのイテレーション数
    for_iter_num = int(for_iter_num_real_initial + for_iter_num_remain_buf +
                       for_remain_iter_num)
    # 得られるデータ数
    print_num = real_bufsize
    if not is_end_iter:
        print_num += (for_iter_num - for_iter_num_print_initial + 1)
    print(f"RI: {real_initial} FITRI: {for_iter_num_print_initial}")
    print(f"FITRII: {for_iter_num_real_initial} is_end_iter: {is_end_iter}")
    print(f"RMNB: {remain_bufsize} FRMNITRB: {for_iter_num_remain_buf}")
    print(f"RB: {real_bufsize} FRNMITR: {for_remain_iter_num}")
    print(f"FITR: {for_iter_num} get_data_num: {print_num}")


def shuffle(data, bufsize, initial, rng, verbose=False):
    startup = True
    initial = min(initial, bufsize)

    for idx, sample in enumerate(data):
        if verbose:
            print(f"idx:{idx}, sample:{sample}, len:{len(buf)}")
            print(f"idx:{idx}, buf:{buf} first")
        if len(buf) < bufsize:
            try:
                tmp = next(data)
                buf.append(tmp)
                if verbose:
                    print(f"idx:{idx}, bufsize:{len(buf)} next:{tmp}")
                    print(f"idx:{idx}, buf:{buf} if first")
            except StopIteration:
                pass
        k = rng.randint(0, len(buf) - 1)
        if verbose:
            print(f"idx:{idx}, k:{k} bufsize:{len(buf)} bufk:{buf[k]} sample:{sample}")
            print(f"idx:{idx}, buf:{buf} second")
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < initial:
            buf.append(sample)
            if verbose:
                print(
                    f"idx:{idx}, k:{k} bufsize:{len(buf)} bufk:{buf[k]} sample:{sample}"
                )
                print(f"idx:{idx}, buf:{buf} if second")
            continue
        if verbose:
            print(f"idx:{idx}, k:{k} bufsize:{len(buf)} bufk:{buf[k]} sample:{sample}")
            print(f"idx:{idx}, buf:{buf} third")
        startup = False
        yield sample
    if verbose:
        print(f"len_buf: {len(buf)} end first for")
    for idx, sample in enumerate(buf):
        if verbose:
            print(f"idx:{idx}, sample:{sample}")
        yield sample


if __name__ == "__main__":
    rng = random.Random()
    pid = os.getpid()
    time_cur = time.time()
    print("main", pid, time_cur)
    # rng.seed((pid, time_cur))
    rng.seed(0)
    buf, bufsize = [], 7
    initial = 100
    data_size = 18
    data_tmp = [i for i in range(data_size)]
    data = iter(data_tmp)
    print("main", data_tmp)
    # test = shuffle(data, bufsize, initial, rng, verbose=False)
    test = shuffle(data, bufsize, initial, rng, verbose=True)
    # print(next(test))
    for idx, t in enumerate(test):
        print("main idx:", idx, t)
    calc_iter_nums(data_size, bufsize, initial)
