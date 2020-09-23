import numpy as np
import scipy.stats as ss
import math
import matplotlib.pyplot as plt


def pde(smp, size, intrvl, k):
    """Probability density estimate"""
    sigma = np.std(smp)
    h = k * 1.06 * sigma * (size ** -0.2)

    pde_y = []
    for x in intrvl:
        sum_k = 0
        for xs in smp:
            sum_k += ss.norm.pdf((x - xs) / h)
        pde_y.append(sum_k / (size * h))
    return pde_y


def draw_pd(smp, size, distrib, name, borders):
    a, b = borders[0], borders[1]
    x = np.arange(a, b, 0.05)
    df_axis = [a, b, -0.01, 0.51]

    fig, ax = plt.subplots(1, 1)
    plt.axis(df_axis)
    k = 1
    plt.plot(x, pde(smp, size, x, k), color='#CC1B92')
    plt.plot(x, distrib.pdf(x), color='black')

    ax.set_title(name + ' n={}'.format(size))
    ax.set_ylabel('f(x)')
    ax.set_xlabel('x')
    ax.grid(axis='y')
    ax.legend(['Ядерная оценка плотности', 'Плотность распределения '])
    plt.show()


def grubbs_test(alpha, sample, mode='both'):
    if sample is None:
        return False
    size = len(sample)

    G, index = -1, 0
    m, s = np.mean(sample), np.std(sample)
    if mode == 'both':
        diffs = [abs(y - m) for y in sample]
        index = np.argmax(diffs)
        G = max(diffs) / s
    elif mode == 'min':
        index = np.argmin(sample)
        G = (m - min(sample)) / s
    elif mode == 'max':
        index = np.argmax(sample)
        G = (max(sample) - m) / s
    if G < 0:
        print('---Error mode---')
        return False

    t_alpha = alpha / (2 * size) if mode == 'both' else alpha / size
    t = ss.t.ppf(1.0 - t_alpha / 2, size - 2)
    G_table = (size - 1) * math.sqrt((t ** 2) / (size - 2 + t ** 2)) / math.sqrt(size)

    if G > G_table:
        sample = np.delete(sample, index)
        print('Индекс обнаруженного выброса: {}'.format(index))
    return sample


def main():
    alpha = 0.2
    size = 100
    outliers = 0.1
    loc = 0
    scale = 1
    distribution = ss.norm
    borders = [-10, 10]

    sample = distribution.rvs(size=size, loc=loc, scale=scale)
    i_o = np.random.choice(size, math.floor(size*outliers))
    print('Индексы сгенерированных выбросов: {}'.format(i_o.tolist()))
    for i in i_o:
        sample[i] = ((-1) ** i) * 100
    # print(sample.tolist())

    before_title = 'До удаления выбросов выбросов'
    draw_pd(sample, len(sample), distribution, before_title, borders)

    modes = ['max', 'min', 'both']
    for mode in modes:
        prev_len, new_len = size + 1, size
        s = np.zeros(len(sample))
        np.copyto(s, sample)
        while new_len < prev_len:
            s = grubbs_test(alpha, s, mode=mode)
            prev_len = new_len
            new_len = len(s)

        after_title = 'После удаления выбросов,\n' + r'$\alpha = $' + '{}, тип теста: {}, '.format(alpha, mode)
        draw_pd(s, len(s), distribution, after_title, borders)


main()
