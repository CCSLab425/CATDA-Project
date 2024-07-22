# Signal Generation
# matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import torch

print('1.下面是一些生成信号并绘制电压、功率（瓦特）和功率（分贝）的代码：')
t = np.linspace(1, 100, 1000)
x_volts = 10*np.sin(t/(2*np.pi))
plt.subplot(3, 1, 1)
plt.plot(t, x_volts)
plt.title('Signal')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')

x_watts = x_volts ** 2  # 计算信号的功率
# print(len(x_watts))
plt.subplot(3, 1, 2)
plt.plot(t, x_watts)
plt.title('Signal Power')
plt.ylabel('Power (W)')
plt.xlabel('Time (s)')

x_db = 10 * np.log10(x_watts)  # 将信号功率W转换为dB
plt.subplot(3, 1, 3)
plt.plot(t, x_db)
plt.title('Signal Power in dB')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.show()



print('2.以下是基于所需信噪比SNR添加高斯白噪声AWGN的示例：')
# snr: noise intensity,like -8,4,2,0,2,4,8
# snr=0 means noise equal to vibration signal
# snr>0 means vibration signal stronger than noise, →∞ means no noise
# snr<0 means noise stronger than vibration signal  →-∞ means no signal
# Adding noise using target SNR

# Set a target SNR
# 设置信噪比
target_snr_db = 4

# Calculate signal power and convert to dB
# 计算信号的功率并将瓦特转换为分贝
sig_avg_watts = np.mean(x_watts)  # 功率-W
sig_avg_db = 10 * np.log10(sig_avg_watts)  # 功率-dB
# Calculate noise according to [2] then convert to watts
# 根据 [2] 计算噪声，然后转换为瓦特
noise_avg_db = sig_avg_db - target_snr_db  # 计算噪声功率dB
noise_avg_watts = 10 ** (noise_avg_db / 10)  # 噪声功率转换为瓦特
# Generate an sample of white noise
mean_noise = 0  # 高斯白噪声均值为0，方差为1
noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
# Noise up the original signal
y_volts = x_volts + noise_volts

# Plot original signal
plt.subplot(3, 1, 1)
plt.plot(t, x_volts)
plt.title('Signal')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')

# Plot signal with noise
plt.subplot(3, 1, 2)
plt.plot(t, y_volts)
plt.title('Signal with noise')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')

# plt.show()
# Plot in dB
y_watts = y_volts ** 2
y_db = 10 * np.log10(y_watts)
plt.subplot(3, 1, 3)
plt.plot(t, 10 * np.log10(y_volts**2))
plt.title('Signal with noise (dB)')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.subplots_adjust(hspace=0.6)  # 调整子图上下间距
plt.show()

print('3.---Python-振动信号加入噪声-代码实现----')
def wgn(x, snr):
    # x: input vibration signal shape (a,b); a:samples number; b samples length
    # snr: noise intensity,like -8,4,2,0,2,4,8
    # snr=0 means noise equal to vibration signal
    # snr>0 means vibration signal stronger than noise, →∞ means no noise
    # snr<0 means noise stronger than vibration signal  →-∞ means no signal
    Ps = np.sum(abs(x)**2, axis=1)/len(x)
    Pn = Ps/(10 ** ((snr/10)))
    row, columns = x.shape
    Pn = np.repeat(Pn.reshape(-1, 1), columns, axis=1)

    noise = np.random.randn(row, columns) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise

