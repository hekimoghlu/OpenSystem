/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#ifdef FIXED_POINT
#include "main_FIX.h"
#else
#include "main_FLP.h"
#endif
#include "tuning_parameters.h"
#include "cpu_support.h"

/*********************************/
/* Initialize Silk Encoder state */
/*********************************/
opus_int silk_init_encoder(
    silk_encoder_state_Fxx          *psEnc,                                 /* I/O  Pointer to Silk FIX encoder state                                           */
    int                              arch                                   /* I    Run-time architecture                                                       */
)
{
    opus_int ret = 0;

    /* Clear the entire encoder state */
    silk_memset( psEnc, 0, sizeof( silk_encoder_state_Fxx ) );

    psEnc->sCmn.arch = arch;

    psEnc->sCmn.variable_HP_smth1_Q15 = silk_LSHIFT( silk_lin2log( SILK_FIX_CONST( VARIABLE_HP_MIN_CUTOFF_HZ, 16 ) ) - ( 16 << 7 ), 8 );
    psEnc->sCmn.variable_HP_smth2_Q15 = psEnc->sCmn.variable_HP_smth1_Q15;

    /* Used to deactivate LSF interpolation, pitch prediction */
    psEnc->sCmn.first_frame_after_reset = 1;

    /* Initialize Silk VAD */
    ret += silk_VAD_Init( &psEnc->sCmn.sVAD );

    return  ret;
}
