# ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

# 1. Augmentation：augmented data for task2 and task3 are also provided in health* and asthma* files;

# function mel(filepath,FRAME_LENGTH,FRAME_INTERVAL)
# 	samps, sr = wavread(filepath)
# 	samps = vec(samps)
# 	frames = powspec(samps, sr; wintime=FRAME_LENGTH, steptime=FRAME_INTERVAL)
# 	energies = log.(sum(frames', dims=2))
# 	fbanks = audspec(frames, sr; nfilts=40, fbtype=:mel)'
# 	fbanks = hcat(fbanks, energies)
# 	# fbank_deltas = deltas(fbanks)
# 	# fbank_deltadeltas = deltas(fbank_deltas)
# 	# features = hcat(fbanks, fbank_deltas, fbank_deltadeltas)
# 	features = hcat(fbanks)
# end
# mel(filepath,FRAME_LENGTH,FRAME_INTERVAL)

# Typical frame sizes in speech processing range from 20 ms to 40 ms with 50% (+/-10%) overlap between consecutive frames.

using DSP
using WAV
using MFCC
using MFCC: fft2barkmx, fft2melmx

function my_powspec(x::Vector{T}, sr::Real=8000.0; wintime=0.025, steptime=0.01, dither=true, window_f::Function) where {T<:AbstractFloat}
	nwin = round(Integer, wintime * sr)
	nstep = round(Integer, steptime * sr)

	nfft = 2 .^ Integer((ceil(log2(nwin))))
	window = window_f(nwin)      # overrule default in specgram which is hamming in Octave
	noverlap = nwin - nstep

	y = spectrogram(x .* (1<<15), nwin, noverlap, nfft=nfft, fs=sr, window=window, onesided=true).power
	## for compability with previous specgram method, remove the last frequency and scale
	y = y[1:end-1, :] ##  * sumabs2(window) * sr / 2
	y .+= dither * nwin / (sum(abs2, window) * sr / 2) ## OK with julia 0.5, 0.6 interpretation as broadcast!

	return y
end

# audspec tested against octave with simple vectors for all fbtypes
function my_audspec(x::Matrix{T}, sr::Real=16000.0; nfilts=ceil(Int, hz2bark(sr/2)), fbtype=:bark,
                 minfreq=0., maxfreq=sr/2, sumpower=true, bwidth=1.0) where {T<:AbstractFloat}
    nfreqs, nframes = size(x)
    nfft = 2(nfreqs-1)
    if fbtype == :bark
        wts = fft2barkmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
    elseif fbtype == :mel
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq)
    elseif fbtype == :htkmel
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq,
                        htkmel=true, constamp=true)
    elseif fbtype == :fcmel
        wts = fft2melmx(nfft, nfilts, sr=sr, width=bwidth, minfreq=minfreq, maxfreq=maxfreq,
                        htkmel=true, constamp=false)
    else
        error("Unknown filterbank type ", fbtype)
    end
    wts = wts[:, 1:nfreqs]
    if sumpower
        return wts * x
    else
        return (wts * sqrt.(x)).^2
    end
end

function my_stft(x::Vector{T}, sr::Real=16000.0; wintime=0.025, steptime=0.01,
              sumpower=false, pre_emphasis=0.97, dither=false, minfreq=0.0, maxfreq=sr/2,
              nbands=20, bwidth=1.0, fbtype=:htkmel,
              usecmp=false, window_f=hamming) where {T<:AbstractFloat}
	if (pre_emphasis != 0)
		x = filt(PolynomialRatio([1., -pre_emphasis], [1.]), x)
	end
	pspec = my_powspec(x, sr, wintime=wintime, steptime=steptime, dither=dither, window_f=window_f)
	aspec = my_audspec(pspec, sr, nfilts=nbands, fbtype=fbtype, minfreq=minfreq, maxfreq=maxfreq, sumpower=sumpower, bwidth=bwidth)
end

merge_channels(samps) = vec(sum(samps, dims=2)/size(samps, 2))

function wav2stft_time_series(filepath, kwargs)
	samps, sr = wavread(filepath)
	samps = merge_channels(samps)

	# wintime = 0.025 # ms
	# steptime = 0.010 # ms
	# fbtype=:mel # [:mel, :htkmel, :fcmel]

	# #window_f = (nwin)->tukey(nwin, 0.25)
	# window_f = hamming

	my_stft(samps, sr; kwargs...)
	# my_stft(samps, sr,
	# 	wintime=wintime, steptime=steptime,
	# 	pre_emphasis=0.97,
	# 	fbtype=:mel,
	# 	nbands=40,
	# 	sumpower=false, dither=false, bwidth=1.0,
	# 	minfreq=0.0, maxfreq=sr/2,
	# 	usecmp=false, window_f=window_f)
end

# wav2stft_time_series(wav_example_filepath)

# DSP.Periodograms.stft(samps, div(length(samps), 8), div(n, 2); onesided=true, nfft=nextfastfft(n), fs=1, window=nothing)

