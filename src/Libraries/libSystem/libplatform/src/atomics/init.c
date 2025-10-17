/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <machine/cpu_capabilities.h>

#include <_simple.h>

#include <platform/string.h>
#include <platform/compat.h>

#include "OSAtomicFifo.h"

__attribute__ ((visibility ("hidden")))
void *COMMPAGE_PFZ_BASE_PTR commpage_pfz_base = 0;

__attribute__ ((visibility ("hidden")))
void
__pfz_setup(const char *apple[])
{
    const char *p = _simple_getenv(apple, "pfz");
	uintptr_t base = 0;
    if (p != NULL) {
        const char *q;

        /* We are given hex starting with 0x */
        if (p[0] != '0' || p[1] != 'x') {
            goto __pfz_setup_clear;
        }

        for (q = p + 2; *q; q++) {
            base <<= 4; // *= 16

            if ('0' <= *q && *q <= '9') {
                base += *q - '0';
            } else if ('a' <= *q && *q <= 'f') {
                base += *q - 'a' + 10;
            } else if ('A' <= *q && *q <= 'F') {
                base += *q - 'A' + 10;
            } else {
                base=0;
                goto __pfz_setup_clear;
            }
        }

__pfz_setup_clear:
        bzero((void *)((uintptr_t)p - 4), strlen(p) + 4);
    }

	if (base != 0) {
		commpage_pfz_base = base;
	}
}

