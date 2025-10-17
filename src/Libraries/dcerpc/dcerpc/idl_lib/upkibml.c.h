/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
**      upkibml.c.h
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      This module contains code to extract information from an IBM
**      long floating number and to initialize an UNPACKED_REAL structure
**      with those bits.
**
**		This module is meant to be used as an include file.
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
**  This module contains code to extract information from an IBM
**  long floating number and to initialize an UNPACKED_REAL structure
**  with those bits.
**
**  See the header files for a description of the UNPACKED_REAL
**  structure.
**
**  A normalized IBM long floating number looks like:
**
**      [0]: Sign bit, 7 exp bits (bias 64), 24 fraction bits
**      [1]: 32 low order fraction bits
**
**      0.0625 <= fraction < 1.0, from 0 to 3 leading zeros
**      to compensate for the hexadecimal exponent.
**
**
**  Implicit parameters:
**
**  	input_value: a pointer to the input parameter.
**
**  	r: an UNPACKED_REAL structure.
**
**  	i: a temporary integer variable.
**
**--
*/

#if (NDR_LOCAL_INT_REP == ndr_c_int_big_endian)

        memcpy(&r[1], input_value, 8);
#if (defined(_IBMR2) && defined(_AIX)) || \
    (defined(__hppa) && defined(__hpux))
        {
        idl_ulong_int tmp;
        tmp = r[1];
        r[1] = r[2];
        r[2] = tmp;
        }
#endif

#else
	memcpy(r, input_value, 8);

	/* Shuffle bytes to little endian format */

	r[2]  = ((r[1] << 24) | (r[1] >> 24));
	r[2] |= ((r[1] << 8) & 0x00FF0000L);
	r[2] |= ((r[1] >> 8) & 0x0000FF00L);
	r[1]  = ((r[0] << 24) | (r[0] >> 24));
	r[1] |= ((r[0] << 8) & 0x00FF0000L);
	r[1] |= ((r[0] >> 8) & 0x0000FF00L);

#endif

	/* Initialize FLAGS and perhaps set NEGATIVE bit */

	r[U_R_FLAGS] = (r[1] >> 31);

	/* Clear sign bit */

	r[1] &= 0x7FFFFFFFL;

	if ((r[1] == 0) && (r[2] == 0)) {

		r[U_R_FLAGS] |= U_R_ZERO;

	} else {

		/* Get unbiased hexadecimal exponent and convert it to binary */

		r[U_R_EXP] = U_R_BIAS + (((r[1] >> 24) - 64) * 4);

		/* Count leading zeros */

		i = 0;
		while (!(r[1] & 0x00800000L)) {
			i += 1;
			if (i > 3)
				break;
			r[1] <<= 1;
		}

		if (i > 3) {

			r[U_R_FLAGS] |= U_R_INVALID;

		} else {

			/* Adjust exponent to compensate for leading zeros */

			r[U_R_EXP] -= i;

			/* Left justify fraction bits */

			r[1] <<= 8;
			i += 8;
			r[1] |= (r[2] >> (32 - i));
			r[2] <<= i;

			/* Clear uninitialized part of unpacked real */

			r[3] = 0;
			r[4] = 0;

		}

	}
