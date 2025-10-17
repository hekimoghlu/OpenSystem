/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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

#include "main_FLP.h"

void silk_LTP_scale_ctrl_FLP(
    silk_encoder_state_FLP          *psEnc,                             /* I/O  Encoder state FLP                           */
    silk_encoder_control_FLP        *psEncCtrl,                         /* I/O  Encoder control FLP                         */
    opus_int                        condCoding                          /* I    The type of conditional coding to use       */
)
{
    opus_int   round_loss;

    if( condCoding == CODE_INDEPENDENTLY ) {
        /* Only scale if first frame in packet */
        round_loss = psEnc->sCmn.PacketLoss_perc * psEnc->sCmn.nFramesPerPacket;
        if ( psEnc->sCmn.LBRR_flag ) {
            /* LBRR reduces the effective loss. In practice, it does not square the loss because
               losses aren't independent, but that still seems to work best. We also never go below 2%. */
            round_loss = 2 + silk_SMULBB( round_loss, round_loss) / 100;
        }
        psEnc->sCmn.indices.LTP_scaleIndex = silk_SMULBB( psEncCtrl->LTPredCodGain, round_loss ) > silk_log2lin( 2900 - psEnc->sCmn.SNR_dB_Q7 );
        psEnc->sCmn.indices.LTP_scaleIndex += silk_SMULBB( psEncCtrl->LTPredCodGain, round_loss ) > silk_log2lin( 3900 - psEnc->sCmn.SNR_dB_Q7 );
    } else {
        /* Default is minimum scaling */
        psEnc->sCmn.indices.LTP_scaleIndex = 0;
    }

    psEncCtrl->LTP_scale = (silk_float)silk_LTPScales_table_Q14[ psEnc->sCmn.indices.LTP_scaleIndex ] / 16384.0f;
}
