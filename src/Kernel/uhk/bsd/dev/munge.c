/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/munge.h>
#include <stdint.h>

/*
 * Refer to comments in bsd/sys/munge.h
 */

static inline __attribute__((always_inline)) void
munge_32_to_64_unsigned(volatile uint64_t *dest, volatile uint32_t *src, int count);

void
munge_w(void *args)
{
	munge_32_to_64_unsigned(args, args, 1);
}

void
munge_ww(void *args)
{
	munge_32_to_64_unsigned(args, args, 2);
}

void
munge_www(void *args)
{
	munge_32_to_64_unsigned(args, args, 3);
}

void
munge_wwww(void *args)
{
	munge_32_to_64_unsigned(args, args, 4);
}

void
munge_wwwww(void *args)
{
	munge_32_to_64_unsigned(args, args, 5);
}

void
munge_wwwwww(void *args)
{
	munge_32_to_64_unsigned(args, args, 6);
}

void
munge_wwwwwww(void *args)
{
	munge_32_to_64_unsigned(args, args, 7);
}

void
munge_wwwwwwww(void *args)
{
	munge_32_to_64_unsigned(args, args, 8);
}

void
munge_wl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwlw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[3] = in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}
void
munge_wwlll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[4] = *(volatile uint64_t*)&in_args[6];
	out_args[3] = *(volatile uint64_t*)&in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwlllll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[6] = *(volatile uint64_t*)&in_args[10];
	out_args[5] = *(volatile uint64_t*)&in_args[8];
	out_args[4] = *(volatile uint64_t*)&in_args[6];
	out_args[3] = *(volatile uint64_t*)&in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwllllll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = *(volatile uint64_t*)&in_args[12];
	out_args[6] = *(volatile uint64_t*)&in_args[10];
	out_args[5] = *(volatile uint64_t*)&in_args[8];
	out_args[4] = *(volatile uint64_t*)&in_args[6];
	out_args[3] = *(volatile uint64_t*)&in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}


void
munge_wwllww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[5] = in_args[7];
	out_args[4] = in_args[6];
	out_args[3] = *(volatile uint64_t*)&in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[2] = in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[3] = in_args[4];
	out_args[2] = in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlwwwll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[6] = *(volatile uint64_t*)&in_args[8];
	out_args[5] = *(volatile uint64_t*)&in_args[6];
	out_args[4] = in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlwwwllw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = in_args[10];
	munge_wlwwwll(args);
}

void
munge_wlwwlwlw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = in_args[10];
	out_args[6] = *(volatile uint64_t*)&in_args[8];
	out_args[5] = in_args[7];
	out_args[4] = *(volatile uint64_t*)&in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[2] = *(volatile uint64_t*)&in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[3] = *(volatile uint64_t*)&in_args[5];
	out_args[2] = *(volatile uint64_t*)&in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlllww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[5] = in_args[8];
	out_args[4] = in_args[7];
	out_args[3] = *(volatile uint64_t*)&in_args[5];
	out_args[2] = *(volatile uint64_t*)&in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wllll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[4] = *(volatile uint64_t*)&in_args[7];
	out_args[3] = *(volatile uint64_t*)&in_args[5];
	out_args[2] = *(volatile uint64_t*)&in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wllww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[4] = in_args[6];
	out_args[3] = in_args[5];
	out_args[2] = *(volatile uint64_t*)&in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wllwwll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[6] = *(volatile uint64_t*)&in_args[9];
	out_args[5] = *(volatile uint64_t*)&in_args[7];
	out_args[4] = in_args[6];
	out_args[3] = in_args[5];
	out_args[2] = *(volatile uint64_t*)&in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwlw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[4] = in_args[5];
	out_args[3] = *(volatile uint64_t*)&in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwlww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[5] = in_args[6];
	out_args[4] = in_args[5];
	out_args[3] = *(volatile uint64_t*)&in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwlwww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[6] = in_args[7];
	out_args[5] = in_args[6];
	out_args[4] = in_args[5];
	out_args[3] = *(volatile uint64_t*)&in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[3] = *(volatile uint64_t*)&in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwlw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[5] = in_args[6];
	out_args[4] = *(volatile uint64_t*)&in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwllww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = in_args[9];
	out_args[6] = in_args[8];
	out_args[5] = *(volatile uint64_t*)&in_args[6];
	out_args[4] = *(volatile uint64_t*)&in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[4] = *(volatile uint64_t*)&in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwwl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[5] = *(volatile uint64_t*)&in_args[5];
	out_args[4] = in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwwlww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = in_args[8];
	out_args[6] = in_args[7];
	out_args[5] = *(volatile uint64_t*)&in_args[5];
	out_args[4] = in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwwllw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = in_args[9];
	out_args[6] = *(volatile uint64_t*)&in_args[7];
	out_args[5] = *(volatile uint64_t*)&in_args[5];
	out_args[4] = in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwwlll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = *(volatile uint64_t*)&in_args[9];
	out_args[6] = *(volatile uint64_t*)&in_args[7];
	out_args[5] = *(volatile uint64_t*)&in_args[5];
	out_args[4] = in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwwwl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[6] = *(volatile uint64_t*)&in_args[6];
	out_args[5] = in_args[5];
	out_args[4] = in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwwwlw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = in_args[8];
	out_args[6] = *(volatile uint64_t*)&in_args[6];
	out_args[5] = in_args[5];
	out_args[4] = in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwwwwwll(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = *(volatile uint64_t*)&in_args[8];
	out_args[6] = *(volatile uint64_t*)&in_args[6];
	out_args[5] = in_args[5];
	out_args[4] = in_args[4];
	out_args[3] = in_args[3];
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wsw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[2] = in_args[2];
	out_args[1] = (int64_t)(int)in_args[1]; /* Sign-extend */
	out_args[0] = in_args[0];
}

void
munge_wws(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[2] = (int64_t)(int)in_args[2]; /* Sign-extend */
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwws(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[3] = (int64_t)(int)in_args[3]; /* Sign-extend */
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}


void
munge_wwwsw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[4] = in_args[4];
	out_args[3] = (int64_t)(int)in_args[3]; /* Sign-extend */
	out_args[2] = in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_llllllll(void *args __unused)
{
	/* Nothing to do, already all 64-bit */
}

void
munge_llllll(void *args __unused)
{
	/* Nothing to do, already all 64-bit */
}

void
munge_llll(void *args __unused)
{
	/* Nothing to do, already all 64-bit */
}

void
munge_lll(void *args __unused)
{
	/* Nothing to do, already all 64-bit */
}

void
munge_ll(void *args __unused)
{
	/* Nothing to do, already all 64-bit */
}

void
munge_l(void *args __unused)
{
	/* Nothing to do, already all 64-bit */
}

void
munge_lw(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[1] = in_args[2];
	out_args[0] = *(volatile uint64_t*)&in_args[0];
}

void
munge_lww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[2] = in_args[3];
	out_args[1] = in_args[2];
	out_args[0] = *(volatile uint64_t*)&in_args[0];
}

void
munge_lwww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[3] = in_args[4];
	out_args[2] = in_args[3];
	out_args[1] = in_args[2];
	out_args[0] = *(volatile uint64_t*)&in_args[0];
}

void
munge_lwwwwwww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[7] = in_args[8];
	out_args[6] = in_args[7];
	out_args[5] = in_args[6];
	out_args[4] = in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = in_args[3];
	out_args[1] = in_args[2];
	out_args[0] = *(volatile uint64_t*)&in_args[0];
}

void
munge_wwlww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[4] = in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwlwww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[5] = in_args[6];
	out_args[4] = in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlwwwl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[5] = *(volatile uint64_t*)&in_args[6];
	out_args[4] = in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wwlwwwl(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[6] = *(volatile uint64_t*)&in_args[7];
	out_args[5] = in_args[6];
	out_args[4] = in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = *(volatile uint64_t*)&in_args[2];
	out_args[1] = in_args[1];
	out_args[0] = in_args[0];
}

void
munge_wlwwlww(void *args)
{
	volatile uint64_t *out_args = (volatile uint64_t*)args;
	volatile uint32_t *in_args = (volatile uint32_t*)args;

	out_args[6] = in_args[8];
	out_args[5] = in_args[7];
	out_args[4] = *(volatile uint64_t*)&in_args[5];
	out_args[3] = in_args[4];
	out_args[2] = in_args[3];
	out_args[1] = *(volatile uint64_t*)&in_args[1];
	out_args[0] = in_args[0];
}

/*
 * Munge array of 32-bit values into an array of 64-bit values,
 * without sign extension.  Note, src and dest can be the same
 * (copies from end of array)
 */
static inline __attribute__((always_inline)) void
munge_32_to_64_unsigned(volatile uint64_t *dest, volatile uint32_t *src, int count)
{
	int i;

	for (i = count - 1; i >= 0; i--) {
		dest[i] = src[i];
	}
}
