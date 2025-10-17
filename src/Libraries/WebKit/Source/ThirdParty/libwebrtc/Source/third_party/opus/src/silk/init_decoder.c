/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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

/************************/
/* Init Decoder State   */
/************************/
opus_int silk_init_decoder(
    silk_decoder_state          *psDec                          /* I/O  Decoder state pointer                       */
)
{
    /* Clear the entire encoder state, except anything copied */
    silk_memset( psDec, 0, sizeof( silk_decoder_state ) );

    /* Used to deactivate LSF interpolation */
    psDec->first_frame_after_reset = 1;
    psDec->prev_gain_Q16 = 65536;
    psDec->arch = opus_select_arch();

    /* Reset CNG state */
    silk_CNG_Reset( psDec );

    /* Reset PLC state */
    silk_PLC_Reset( psDec );

    return(0);
}

