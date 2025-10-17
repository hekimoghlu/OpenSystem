/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
/*
**
**  NAME:
**
**      pkvaxg.c.h
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      This module contains code to extract information from an
**      UNPACKED_REAL structure and to create a VAX g_floating number
**      with those bits.
**
**              This module is meant to be used as an include file.
**
**  VERSION: DCE 1.0
**
*/
#if HAVE_CONFIG_H
#include <config.h>
#endif

/*
**++
**  Functional Description:
**
**  This module contains code to extract information from an
**  UNPACKED_REAL structure and to create a VAX g_floating number
**  with those bits.
**
**  See the header files for a description of the UNPACKED_REAL
**  structure.
**
**  A VAX g_floating number in (16 bit words) looks like:
**
**      [0]: Sign bit, 11 exp bits (bias 1024), 4 fraction bits
**      [1]: 16 more fraction bits
**      [2]: 16 more fraction bits
**      [3]: 16 more fraction bits
**
**      0.5 <= fraction < 1.0, MSB implicit
**
**
**  Implicit parameters:
**
**      options: a word of flags, see include files.
**
**      output_value: a pointer to the input parameter.
**
**      r: an UNPACKED_REAL structure.
**
**--
*/

if (r[U_R_FLAGS] & U_R_UNUSUAL) {

        if (r[U_R_FLAGS] & U_R_ZERO)

                memcpy(output_value, VAX_G_ZERO, 8);

        else if (r[U_R_FLAGS] & U_R_INFINITY) {

                memcpy(output_value, VAX_G_INVALID, 8);
                if (r[U_R_FLAGS] & U_R_NEGATIVE) {
                        DCETHREAD_RAISE(dcethread_aritherr_e);    /* Negative infinity */
                } else {
                        DCETHREAD_RAISE(dcethread_aritherr_e);    /* Positive infinity */
                }

        } else if (r[U_R_FLAGS] & U_R_INVALID) {

                memcpy(output_value, VAX_G_INVALID, 8);
                DCETHREAD_RAISE(dcethread_aritherr_e);    /* Invalid value */

        }

} else {

        round_bit_position = 53;

#include "round.c.h"

        if (r[U_R_EXP] < (U_R_BIAS - 1023)) {

                /* Underflow */

                memcpy(output_value, VAX_G_ZERO, 8);
                if (options & CVT_C_ERR_UNDERFLOW) {
                        DCETHREAD_RAISE(dcethread_fltund_e);  /* Underflow */
                }

        } else if (r[U_R_EXP] > (U_R_BIAS + 1023)) {

                /* Overflow */

                if (options & CVT_C_TRUNCATE) {

                        if (r[U_R_FLAGS] & U_R_NEGATIVE)
                                memcpy(output_value, VAX_G_NEG_HUGE, 8);
                        else
                                memcpy(output_value, VAX_G_POS_HUGE, 8);

                } else if ((options & CVT_C_ROUND_TO_POS)
                                        && (r[U_R_FLAGS] & U_R_NEGATIVE)) {

                                memcpy(output_value, VAX_G_NEG_HUGE, 8);

                } else if ((options & CVT_C_ROUND_TO_NEG)
                                        && !(r[U_R_FLAGS] & U_R_NEGATIVE)) {

                                memcpy(output_value, VAX_G_POS_HUGE, 8);

                } else {

                        memcpy(output_value, VAX_G_INVALID, 8);

                }

                DCETHREAD_RAISE(dcethread_fltovf_e);  /* Overflow */

        } else {

                /* Adjust bias of exponent */

                r[U_R_EXP] -= (U_R_BIAS - 1024);

                /* Make room for exponent and sign bit */

                r[2] >>= 11;
                r[2] |= (r[1] << 21);
                r[1] >>= 11;

                /* Clear implicit bit */

                r[1] &= 0x000FFFFFL;

                /* OR in exponent and sign bit */

                r[1] |= (r[U_R_EXP] << 20);
                r[1] |= (r[U_R_FLAGS] << 31);

                /* Adjust for VAX 16 bit floating format */

                r[1] = ((r[1] << 16) | (r[1] >> 16));
                r[2] = ((r[2] << 16) | (r[2] >> 16));

                memcpy(output_value, &r[1], 8);

        }

}
