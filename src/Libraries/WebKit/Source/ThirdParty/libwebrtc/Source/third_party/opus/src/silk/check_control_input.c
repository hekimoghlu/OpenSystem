/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main.h"
#include "control.h"
#include "errors.h"

/* Check encoder control struct */
opus_int check_control_input(
    silk_EncControlStruct        *encControl                    /* I    Control structure                           */
)
{
    celt_assert( encControl != NULL );

    if( ( ( encControl->API_sampleRate            !=  8000 ) &&
          ( encControl->API_sampleRate            != 12000 ) &&
          ( encControl->API_sampleRate            != 16000 ) &&
          ( encControl->API_sampleRate            != 24000 ) &&
          ( encControl->API_sampleRate            != 32000 ) &&
          ( encControl->API_sampleRate            != 44100 ) &&
          ( encControl->API_sampleRate            != 48000 ) ) ||
        ( ( encControl->desiredInternalSampleRate !=  8000 ) &&
          ( encControl->desiredInternalSampleRate != 12000 ) &&
          ( encControl->desiredInternalSampleRate != 16000 ) ) ||
        ( ( encControl->maxInternalSampleRate     !=  8000 ) &&
          ( encControl->maxInternalSampleRate     != 12000 ) &&
          ( encControl->maxInternalSampleRate     != 16000 ) ) ||
        ( ( encControl->minInternalSampleRate     !=  8000 ) &&
          ( encControl->minInternalSampleRate     != 12000 ) &&
          ( encControl->minInternalSampleRate     != 16000 ) ) ||
          ( encControl->minInternalSampleRate > encControl->desiredInternalSampleRate ) ||
          ( encControl->maxInternalSampleRate < encControl->desiredInternalSampleRate ) ||
          ( encControl->minInternalSampleRate > encControl->maxInternalSampleRate ) ) {
        celt_assert( 0 );
        return SILK_ENC_FS_NOT_SUPPORTED;
    }
    if( encControl->payloadSize_ms != 10 &&
        encControl->payloadSize_ms != 20 &&
        encControl->payloadSize_ms != 40 &&
        encControl->payloadSize_ms != 60 ) {
        celt_assert( 0 );
        return SILK_ENC_PACKET_SIZE_NOT_SUPPORTED;
    }
    if( encControl->packetLossPercentage < 0 || encControl->packetLossPercentage > 100 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_LOSS_RATE;
    }
    if( encControl->useDTX < 0 || encControl->useDTX > 1 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_DTX_SETTING;
    }
    if( encControl->useCBR < 0 || encControl->useCBR > 1 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_CBR_SETTING;
    }
    if( encControl->useInBandFEC < 0 || encControl->useInBandFEC > 1 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_INBAND_FEC_SETTING;
    }
    if( encControl->nChannelsAPI < 1 || encControl->nChannelsAPI > ENCODER_NUM_CHANNELS ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if( encControl->nChannelsInternal < 1 || encControl->nChannelsInternal > ENCODER_NUM_CHANNELS ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if( encControl->nChannelsInternal > encControl->nChannelsAPI ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_NUMBER_OF_CHANNELS_ERROR;
    }
    if( encControl->complexity < 0 || encControl->complexity > 10 ) {
        celt_assert( 0 );
        return SILK_ENC_INVALID_COMPLEXITY_SETTING;
    }

    return SILK_NO_ERROR;
}
