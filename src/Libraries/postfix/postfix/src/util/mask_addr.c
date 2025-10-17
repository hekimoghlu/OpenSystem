/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
/* System library. */

#include <sys_defs.h>
#include <limits.h>			/* CHAR_BIT */

/* Utility library. */

#include <msg.h>
#include <mask_addr.h>

/* mask_addr - mask off a variable-length address */

void    mask_addr(unsigned char *addr_bytes,
		          unsigned addr_byte_count,
		          unsigned network_bits)
{
    unsigned char *p;

    if (network_bits > addr_byte_count * CHAR_BIT)
	msg_panic("mask_addr: address byte count %d too small for bit count %d",
		  addr_byte_count, network_bits);

    p = addr_bytes + network_bits / CHAR_BIT;
    network_bits %= CHAR_BIT;

    if (network_bits != 0)
	*p++ &= ~0U << (CHAR_BIT - network_bits);

    while (p < addr_bytes + addr_byte_count)
	*p++ = 0;
}
