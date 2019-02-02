'''   
    Created in: 31 Oct 2016
    Python Version: 3
    License: GPLv3

    Author: Jeova Pereira
    Github: @jeovazero
    Email: contato@jeova.ninja
'''

import numpy as np
from scipy.stats import multivariate_normal
from plotly.offline import plot
from plotly.graph_objs import Scatter, Box
import plotly.graph_objs as go
from plotly import tools


ln = 0
Po = .00064
alpha = 0
alph = alpha/(4.343)
gamma = 0.003
to = 125e-12
C = -2
b2 = -20e-27
Ld = (to**2)/np.absolute(b2)
pi = 3.1415926535
Ao = np.sqrt(Po)


tau = np.arange(-4096e-12, 4095e-12, 1e-12)
dt = 1e-12
rel_error = 1e-5
h = 1000


op_pulse = [[0 for y in tau] for x in np.arange(0.1, 1.51, 0.1)]
pbratio = [0  for x in np.arange(0.1, 1.51, 0.1)]
phadisp = [0 for x in np.arange(0.1, 1.51, 0.1)]


uaux = Ao*np.exp(-((1+1j*(-C))/2.0)*(tau/to)**2)

for ii in np.arange(0.1, 1.51, 0.1):
    z = ii * Ld
    u = uaux[:]
    l = np.max(u.shape)
    fwhml = np.nonzero(np.absolute(u) > np.absolute(np.max(np.real(u))/2.0))  
    fwhml = len(fwhml[0])
    dw = 1.0/float(l)/dt*2.0*pi

    w = dw*np.arange(-1*l/2.0, l/2.0, 1)
    w = np.asarray(w)
    w = np.fft.fftshift(w)

    u = np.asarray(u)
    u = np.fft.fftshift(u)

    spectrum = np.fft.fft(np.fft.fftshift(u))

    for jj in np.arange(h, z+1, h):
        spectrum = spectrum*np.exp(-alph*(h/2.0)+1j*b2/2.0*(np.power(w, 2))*(h/2.0))
        f = np.fft.ifft(spectrum)
        f = f*np.exp(1j*gamma*np.power(np.absolute(f), 2)*(h))
        spectrum = np.fft.fft(f)
        spectrum = spectrum*np.exp(-alph*(h/2.0)+1j*b2/2.0*(np.power(w, 2)*(h/2.0)))

    f = np.fft.ifft(spectrum)
    op_pulse[ln] = np.absolute(f)
    fwhm = np.nonzero(np.absolute(f) > np.absolute(np.max(np.real(f))/2.0))
    fwhm = len(fwhm[0])
    ratio = float(fwhm)/fwhml
    pbratio[ln] = ratio
    im = np.absolute(np.imag(f))
    re = np.absolute(np.real(f))
    div = np.dot(im, np.linalg.pinv([re]))
    dd = np.degrees(np.arctan(div))
    phadisp[ln] = dd[0]
    print("> %d / 14 " % ln)
    ln = ln + 1

# Plots
print("\n\n> Plotting...")
trace_pulse_evolution = go.Surface(z=op_pulse, colorscale='Jet')
trace_pulse_broad = go.Scatter(y=pbratio[0:ln], x=np.arange(1,ln+1,1))
trace_phase_change = go.Scatter(y=phadisp[0:ln], x=np.arange(1,ln+1,1))
layout_input_pulse = go.Layout(
    autosize = False,
    width=500,
    height=400,
    title='Input Pulse',
    xaxis=dict(
        title='Time',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Amplitude',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

layout_pulse_evolution = go.Layout(
    autosize = False,
    width=800,
    height=800,
    title='Pulse Evolution',
    scene=go.Scene(
        xaxis=go.XAxis(title='Time'),
        yaxis=go.YAxis(title='Distance'),
        zaxis=go.ZAxis(title='Amplitude'))
)

trace_input_pulse = go.Scatter(y=np.absolute(uaux))

pulse_evolution = go.Figure(data=[trace_pulse_evolution], layout=layout_pulse_evolution)

input_pulse = go.Figure(data=[trace_input_pulse], layout=layout_input_pulse)

plot([trace_phase_change], filename='./phase_change.html')
plot([trace_pulse_broad], filename='./pulse_broadening.html')
plot(pulse_evolution, filename='./pulse_evolution.html')
plot(input_pulse, filename='./input_pulse.html')
