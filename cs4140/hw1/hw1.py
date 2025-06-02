import random
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def cdf(data):
    """Calculates the empirical CDF of a given dataset."""
    # sorted the data
    sorted_data = sorted(data)
    # Calculate the length of the data
    n = len(sorted_data)
    # Create a list of CDF values
    cdf_values = []
    for i, x in enumerate(sorted_data):
        cdf_values.append((i+1) / n)

    return sorted_data, cdf_values

def plot_times(x, y, num_ys):
    colors = ["b", "g", "r", "y", "m", "c", "k"]
    for i in range(0, len(num_ys)):
        plt.plot(x, y[i], label=f"M of {num_ys[i]}", marker='o', linestyle='-', color=colors[i])
    plt.xlabel("N")
    plt.ylabel("Times in seconds")
    plt.title("Time it Takes for N domain and M iterations")
    plt.legend()
    plt.show()

def plot_cdf(data):
    x, y = cdf(data=data)
    plt.step(x, y, where="post")
    plt.xlabel("Data")
    plt.ylabel("CDF")
    plt.title("Empirical CDF")
    plt.show()

def get_number(limit) -> int:
    return random.randint(0,limit - 1)

def two_same_numbers(limit=5000) -> int:
    set_o_nums = np.zeros(limit)
    flag = True
    count = 0
    while flag:
        num = get_number(limit)
        if set_o_nums[num] == 1:
            flag = False
        else:
            set_o_nums[num] = 1
        count += 1
    
    return count

def count_to_fill_array(domain) -> int:
    count = 0
    increases = 0
    set_o_nums = np.zeros(domain)
    while True:
        num = get_number(domain)             
        if set_o_nums[num] == 0:
            set_o_nums[num] = 1
            increases +=1
        count += 1
        if increases == domain:
            break
    return count

if __name__ == "__main__":
    
    print(f"Birthday: {two_same_numbers()}")
    print(f"Coupons: {count_to_fill_array(300)}")
    part = sys.argv[1]
    if part == "1":
        M = [10_000, 1_000, 300]
        N = [1_000_000, 750_000, 500_000, 250_000, 100_000, 50_000, 10_000, 5_000]
        ns = N[::-1]
        overall = []
        for m in M:
            times = []
            for n in N:
                k_vals = np.zeros(m)
                start = time.time()
                for i in range(0, m):
                    k_vals[i] = two_same_numbers()
                end = time.time()

                elapsed_time = end - start
                times.append(elapsed_time)
                print(f"Took: {elapsed_time} for M:{m} and N:{n}")
                print(f"For M = {m} and N = {n} the mean = {np.mean(k_vals)}")

            temp = times[::-1]
            overall.append(temp)

        plot_times(ns, overall, M)
        plot_cdf(k_vals)

    elif part == "2":
        N = [20_000, 15_000, 10_000, 5_000, 1_000, 300]
        M = [5_000, 2_000, 400]
        ns = N[::-1]
        overall = []
        for m in M:
            times = []
            for n in N:
                coupons = np.zeros(m)
                increases = 0
                start = time.time()
                for i in range (0, m):
                    coupons[i] = count_to_fill_array(n)
                end = time.time()
                
                elapsed_time = end - start
                times.append(elapsed_time)
                print(f"Took: {elapsed_time} for M:{m} and N:{n}")
                print(f"For M = {m} and N = {n} the mean = {np.mean(coupons)}")
                
            temp = times[::-1]
            overall.append(temp)
            
        plot_times(ns, overall, M)    
        plot_cdf(coupons)
        