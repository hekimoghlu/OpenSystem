/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include "sudoers.h"

/*
 * Derived from code with the following declaration:
 *	PUBLIC DOMAIN - Jon Mayo - November 13, 2003 
 */

static const unsigned char base64dec_tab[256]= {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
     52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,  0,255,255,
    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,
    255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};

/*
 * Decode a NUL-terminated string in base64 format and store the
 * result in dst.
 */
size_t
base64_decode(const char *in, unsigned char *out, size_t out_size)
{
    unsigned char *out_end = out + out_size;
    const unsigned char *out0 = out;
    unsigned int rem, v;
    debug_decl(base64_decode, SUDOERS_DEBUG_MATCH);

    for (v = 0, rem = 0; *in != '\0' && *in != '='; in++) {
	unsigned char ch = base64dec_tab[(unsigned char)*in];
	if (ch == 255)
	    debug_return_size_t((size_t)-1);
	v = (v << 6) | ch;
	rem += 6;
	if (rem >= 8) {
	    rem -= 8;
	    if (out >= out_end)
		debug_return_size_t((size_t)-1);
	    *out++ = (v >> rem) & 0xff;
	}
    }
    if (rem >= 8) {
	    if (out >= out_end)
		debug_return_size_t((size_t)-1);
	    *out++ = (v >> rem) & 0xff;
    }
    debug_return_size_t((size_t)(out - out0));
}
