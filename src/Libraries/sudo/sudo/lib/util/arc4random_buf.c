/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

#ifndef HAVE_ARC4RANDOM_BUF

#include <stdlib.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif

#include "sudo_compat.h"
#include "sudo_rand.h"

#define minimum(a, b) ((a) < (b) ? (a) : (b))

/*
 * Call arc4random() repeatedly to fill buf with n bytes of random data.
 */
void
sudo_arc4random_buf(void *buf, size_t n)
{
	char *cp = buf;

	while (n != 0) {
		size_t m = minimum(n, 4);
		uint32_t val = arc4random();

		switch (m) {
		case 4:
			*cp++ = (val >> 24) & 0xff;
			FALLTHROUGH;
		case 3:
			*cp++ = (val >> 16) & 0xff;
			FALLTHROUGH;
		case 2:
			*cp++ = (val >> 8) & 0xff;
			FALLTHROUGH;
		case 1:
			*cp++ = val & 0xff;
			break;
		}
		n -= m;
	}
}

#endif /* HAVE_ARC4RANDOM_BUF */
