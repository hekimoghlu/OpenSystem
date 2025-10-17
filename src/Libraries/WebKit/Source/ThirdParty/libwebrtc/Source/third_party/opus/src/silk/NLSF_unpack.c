/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

/* Unpack predictor values and indices for entropy coding tables */
void silk_NLSF_unpack(
          opus_int16            ec_ix[],                        /* O    Indices to entropy tables [ LPC_ORDER ]     */
          opus_uint8            pred_Q8[],                      /* O    LSF predictor [ LPC_ORDER ]                 */
    const silk_NLSF_CB_struct   *psNLSF_CB,                     /* I    Codebook object                             */
    const opus_int              CB1_index                       /* I    Index of vector in first LSF codebook       */
)
{
    opus_int   i;
    opus_uint8 entry;
    const opus_uint8 *ec_sel_ptr;

    ec_sel_ptr = &psNLSF_CB->ec_sel[ CB1_index * psNLSF_CB->order / 2 ];
    for( i = 0; i < psNLSF_CB->order; i += 2 ) {
        entry = *ec_sel_ptr++;
        ec_ix  [ i     ] = silk_SMULBB( silk_RSHIFT( entry, 1 ) & 7, 2 * NLSF_QUANT_MAX_AMPLITUDE + 1 );
        pred_Q8[ i     ] = psNLSF_CB->pred_Q8[ i + ( entry & 1 ) * ( psNLSF_CB->order - 1 ) ];
        ec_ix  [ i + 1 ] = silk_SMULBB( silk_RSHIFT( entry, 5 ) & 7, 2 * NLSF_QUANT_MAX_AMPLITUDE + 1 );
        pred_Q8[ i + 1 ] = psNLSF_CB->pred_Q8[ i + ( silk_RSHIFT( entry, 4 ) & 1 ) * ( psNLSF_CB->order - 1 ) + 1 ];
    }
}

