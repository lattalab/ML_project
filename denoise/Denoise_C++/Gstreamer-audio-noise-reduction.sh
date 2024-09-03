#!/bin/sh
# ref: https://github.com/nicolargo/gstpipelinearena/blob/master/audio-noise-reduction.sh

SOURCE="filesrc location=./CH_fake_noise.wav ! decodebin ! audioconvert"
TIME=5
CUTOFF=500  # filter's cutting frequency
SAMPLE_RATE=22050

# Output file paths
OUTPUT_SANS="output_sans.wav"
OUTPUT_AVEC_CHEB="output_avec_cheb.wav"
OUTPUT_AVEC_SINC="output_avec_sinc.wav"

# Sans
echo "Sans filtrage"
timeout $TIME gst-launch-1.0 $SOURCE \
	! audioresample ! audio/x-raw,rate=$SAMPLE_RATE \
	! audioconvert ! wavenc ! filesink location=$OUTPUT_SANS

# Avec
echo "Avec filtrage audiocheblimit"
echo "Supprime les fréquence inférieure à $CUTOFF"
timeout $TIME gst-launch-1.0 $SOURCE \
	! audiocheblimit mode=1 cutoff=$CUTOFF ! audiodynamic ! audioconvert noise-shaping=4 \
	! audioresample ! audio/x-raw,rate=$SAMPLE_RATE \
	! audioconvert ! wavenc ! filesink location=$OUTPUT_AVEC_CHEB

# Avec
echo "Avec filtrage audiowsinclimit"
timeout $TIME gst-launch-1.0 $SOURCE \
	! audiowsinclimit mode=1 length=200 cutoff=$CUTOFF ! audiodynamic ! audioconvert noise-shaping=4 \
	! audioresample ! audio/x-raw,rate=$SAMPLE_RATE \
	! audioconvert ! wavenc ! filesink location=$OUTPUT_AVEC_SINC

