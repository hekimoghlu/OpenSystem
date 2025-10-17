/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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

static const unsigned char base64enc_tab[64] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t
base64_encode(const unsigned char *in, size_t in_len, char *out, size_t out_len)
{
    size_t ii, io;
    unsigned int rem, v;
    debug_decl(base64_encode, SUDOERS_DEBUG_MATCH);

    for (io = 0, ii = 0, v = 0, rem = 0; ii < in_len; ii++) {
	unsigned char ch = in[ii];
	v = (v << 8) | ch;
	rem += 8;
	while (rem >= 6) {
	    rem -= 6;
	    if (io >= out_len)
		debug_return_size_t((size_t)-1); /* truncation is failure */
	    out[io++] = base64enc_tab[(v >> rem) & 63];
	}
    }
    if (rem != 0) {
	v <<= (6 - rem);
	if (io >= out_len)
	    debug_return_size_t((size_t)-1); /* truncation is failure */
	out[io++] = base64enc_tab[v&63];
    }
    while (io & 3) {
	if (io >= out_len)
	    debug_return_size_t((size_t)-1); /* truncation is failure */
	out[io++] = '=';
    }
    if (io >= out_len)
	debug_return_size_t((size_t)-1); /* no room for NUL terminator */
    out[io] = '\0';
    debug_return_size_t(io);
}
