/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef SILK_CONTROL_H
#define SILK_CONTROL_H

#include "typedef.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Decoder API flags */
#define FLAG_DECODE_NORMAL                      0
#define FLAG_PACKET_LOST                        1
#define FLAG_DECODE_LBRR                        2

/***********************************************/
/* Structure for controlling encoder operation */
/***********************************************/
typedef struct {
    /* I:   Number of channels; 1/2                                                         */
    opus_int32 nChannelsAPI;

    /* I:   Number of channels; 1/2                                                         */
    opus_int32 nChannelsInternal;

    /* I:   Input signal sampling rate in Hertz; 8000/12000/16000/24000/32000/44100/48000   */
    opus_int32 API_sampleRate;

    /* I:   Maximum internal sampling rate in Hertz; 8000/12000/16000                       */
    opus_int32 maxInternalSampleRate;

    /* I:   Minimum internal sampling rate in Hertz; 8000/12000/16000                       */
    opus_int32 minInternalSampleRate;

    /* I:   Soft request for internal sampling rate in Hertz; 8000/12000/16000              */
    opus_int32 desiredInternalSampleRate;

    /* I:   Number of samples per packet in milliseconds; 10/20/40/60                       */
    opus_int payloadSize_ms;

    /* I:   Bitrate during active speech in bits/second; internally limited                 */
    opus_int32 bitRate;

    /* I:   Uplink packet loss in percent (0-100)                                           */
    opus_int packetLossPercentage;

    /* I:   Complexity mode; 0 is lowest, 10 is highest complexity                          */
    opus_int complexity;

    /* I:   Flag to enable in-band Forward Error Correction (FEC); 0/1                      */
    opus_int useInBandFEC;

    /* I:   Flag to actually code in-band Forward Error Correction (FEC) in the current packet; 0/1 */
    opus_int LBRR_coded;

    /* I:   Flag to enable discontinuous transmission (DTX); 0/1                            */
    opus_int useDTX;

    /* I:   Flag to use constant bitrate                                                    */
    opus_int useCBR;

    /* I:   Maximum number of bits allowed for the frame                                    */
    opus_int maxBits;

    /* I:   Causes a smooth downmix to mono                                                 */
    opus_int toMono;

    /* I:   Opus encoder is allowing us to switch bandwidth                                 */
    opus_int opusCanSwitch;

    /* I: Make frames as independent as possible (but still use LPC)                        */
    opus_int reducedDependency;

    /* O:   Internal sampling rate used, in Hertz; 8000/12000/16000                         */
    opus_int32 internalSampleRate;

    /* O: Flag that bandwidth switching is allowed (because low voice activity)             */
    opus_int allowBandwidthSwitch;

    /* O:   Flag that SILK runs in WB mode without variable LP filter (use for switching between WB/SWB/FB) */
    opus_int inWBmodeWithoutVariableLP;

    /* O:   Stereo width */
    opus_int stereoWidth_Q14;

    /* O:   Tells the Opus encoder we're ready to switch                                    */
    opus_int switchReady;

    /* O: SILK Signal type */
    opus_int signalType;

    /* O: SILK offset (dithering) */
    opus_int offset;
} silk_EncControlStruct;

/**************************************************************************/
/* Structure for controlling decoder operation and reading decoder status */
/**************************************************************************/
typedef struct {
    /* I:   Number of channels; 1/2                                                         */
    opus_int32 nChannelsAPI;

    /* I:   Number of channels; 1/2                                                         */
    opus_int32 nChannelsInternal;

    /* I:   Output signal sampling rate in Hertz; 8000/12000/16000/24000/32000/44100/48000  */
    opus_int32 API_sampleRate;

    /* I:   Internal sampling rate used, in Hertz; 8000/12000/16000                         */
    opus_int32 internalSampleRate;

    /* I:   Number of samples per packet in milliseconds; 10/20/40/60                       */
    opus_int payloadSize_ms;

    /* O:   Pitch lag of previous frame (0 if unvoiced), measured in samples at 48 kHz      */
    opus_int prevPitchLag;
} silk_DecControlStruct;

#ifdef __cplusplus
}
#endif

#endif
