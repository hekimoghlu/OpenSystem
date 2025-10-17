/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
**      upkieeet.c.h
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      This module contains code to extract information from a IEEE
**      double floating number and to initialize an UNPACKED_REAL structure
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
**  This module contains code to extract information from a IEEE
**  double floating number and to initialize an UNPACKED_REAL structure
**  with those bits.
**
**  See the header files for a description of the UNPACKED_REAL
**  structure.
**
**  A normalized IEEE double precision floating number looks like:
**
**      [0]: 32 low order fraction bits
**      [1]: Sign bit, 11 exp bits (bias 1023), 20 fraction bits
**
**      1.0 <= fraction < 2.0, MSB implicit
**
**  For more details see "Mips R2000 Risc Architecture"
**  by Gerry Kane, page 6-8 or ANSI/IEEE Std 754-1985.
**
**
**  Implicit parameters:
**
**  	options: a word of flags, see include files.
**
**  	input_value: a pointer to the input parameter.
**
**  	r: an UNPACKED_REAL structure.
**
**  	i: a temporary integer variable.
**
**--
*/

	memcpy(r, input_value, 8);

#if (NDR_LOCAL_INT_REP == ndr_c_int_big_endian)
	memcpy(&r[1], input_value, 8);
#else
	if (options & CVT_C_BIG_ENDIAN) {

		/* Shuffle bytes to little endian format */

		r[2]  = ((r[1] << 24) | (r[1] >> 24));
		r[2] |= ((r[1] << 8) & 0x00FF0000L);
		r[2] |= ((r[1] >> 8) & 0x0000FF00L);
		r[1]  = ((r[0] << 24) | (r[0] >> 24));
		r[1] |= ((r[0] << 8) & 0x00FF0000L);
		r[1] |= ((r[0] >> 8) & 0x0000FF00L);

	} else {

		r[2] = r[0];

	}
#endif

	/* Initialize FLAGS and perhaps set NEGATIVE bit */

	r[U_R_FLAGS] = (r[1] >> 31);

	/* Extract biased exponent */

	r[U_R_EXP] = (r[1] >> 20) & 0x000007FFL;

	/* Check for denormalized values */

	if (r[U_R_EXP] == 0) {

		/* Clear sign bit */

		r[1] &= 0x7FFFFFFFL;

		/* If fraction is non-zero then normalize it */

		if (r[1]) {

			/* Count leading zeros in fraction */

			i = 0;
			while (!(r[1] & 0x00080000L)) {
				r[1] <<= 1;
				i += 1;
			}

			/* Adjust exponent and normalize fraction */

			r[U_R_EXP] = U_R_BIAS - 1022 - i;
			r[1] <<= 12;
			i += 12;
			r[1] |= (r[2] >> (32 - i));
			r[2] <<= i;

			/* Clear uninitialized part of unpacked real */

			r[3] = 0;
			r[4] = 0;

		} else if (r[2]) {

			r[1] = r[2];

			/* Count leading zeros in fraction */

			i = 20;
			while (!(r[1] & 0x80000000L)) {
				r[1] <<= 1;
				i += 1;
			}

			/* Adjust exponent */

			r[U_R_EXP] = U_R_BIAS - 1022 - i;

			/* Clear uninitialized part of unpacked real */

			r[2] = 0;
			r[3] = 0;
			r[4] = 0;

		} else {

			r[U_R_FLAGS] |= U_R_ZERO;

		}

	/* Check for NANs and INFINITIES */

	} else if (r[U_R_EXP] == 2047) {

		/* Clear sign and exponent */

		r[1] &= 0x000FFFFFL;

		if (r[1] | r[2])
			r[U_R_FLAGS] |= U_R_INVALID;
		else
			r[U_R_FLAGS] |= U_R_INFINITY;

	} else {

		/* Adjust exponent bias */

		r[U_R_EXP] += (U_R_BIAS - 1022);

		/* Set hidden bit */

		r[1] |= 0x00100000L;

		/* Left justify fraction bits */

		r[1] <<= 11;
		r[1] |= (r[2] >> 21);
		r[2] <<= 11;

		/* Clear uninitialized part of unpacked real */

		r[3] = 0;
		r[4] = 0;

	}
