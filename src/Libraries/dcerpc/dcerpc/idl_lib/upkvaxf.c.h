/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
**      upkvaxf.c.h
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      This module contains code to extract information from a VAX
**      f_floating number and to initialize an UNPACKED_REAL structure
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
**  This module contains code to extract information from a VAX
**  f_floating number and to initialize an UNPACKED_REAL structure
**  with those bits.
**
**  See the header files for a description of the UNPACKED_REAL
**  structure.
**
**  A VAX f_floating number in (16 bit words) looks like:
**
**      [0]: Sign bit, 8 exp bits (bias 128), 7 fraction bits
**      [1]: 16 more fraction bits
**
**      0.5 <= fraction < 1.0, MSB implicit
**
**
**  Implicit parameters:
**
**  	input_value: a pointer to the input parameter.
**
**  	r: an UNPACKED_REAL structure
**
**--
*/

	memcpy(&r[1], input_value, 4);

	/* Initialize FLAGS and perhaps set NEGATIVE bit */

	r[U_R_FLAGS] = (r[1] >> 15) & U_R_NEGATIVE;

	/* Extract VAX biased exponent */

	r[U_R_EXP] = (r[1] >> 7) & 0x000000FFL;

	if (r[U_R_EXP] == 0) {

		if (r[U_R_FLAGS])
			r[U_R_FLAGS] |= U_R_INVALID;
		else
			r[U_R_FLAGS] = U_R_ZERO;

	} else {

		/* Adjust for VAX 16 bit floating format */

		r[1] = ((r[1] << 16) | (r[1] >> 16));

		/* Add unpacked real bias and subtract VAX bias */

		r[U_R_EXP] += (U_R_BIAS - 128);

		/* Set hidden bit */

		r[1] |= 0x00800000L;

		/* Left justify fraction bits */

		r[1] <<= 8;

		/* Clear uninitialized parts for unpacked real */

		r[2] = 0;
		r[3] = 0;
		r[4] = 0;

	}
