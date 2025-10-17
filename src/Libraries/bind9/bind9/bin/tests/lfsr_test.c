/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
/* $Id: lfsr_test.c,v 1.16 2007/06/19 23:46:59 tbox Exp $ */

/*! \file */
#include <config.h>

#include <stdio.h>

#include <isc/lfsr.h>
#include <isc/print.h>
#include <isc/util.h>

isc_uint32_t state[1024 * 64];

int
main(int argc, char **argv) {
	isc_lfsr_t lfsr1, lfsr2;
	int i;
	isc_uint32_t temp;

	UNUSED(argc);
	UNUSED(argv);

	/*
	 * Verify that returned values are reproducable.
	 */
	isc_lfsr_init(&lfsr1, 0, 32, 0x80000057U, 0, NULL, NULL);
	for (i = 0; i < 32; i++) {
		isc_lfsr_generate(&lfsr1, &state[i], 4);
		printf("lfsr1:  state[%2d] = %08x\n", i, state[i]);
	}
	isc_lfsr_init(&lfsr1, 0, 32, 0x80000057U, 0, NULL, NULL);
	for (i = 0; i < 32; i++) {
		isc_lfsr_generate(&lfsr1, &temp, 4);
		if (state[i] != temp)
			printf("lfsr1:  state[%2d] = %08x, "
			       "but new state is %08x\n",
			       i, state[i], temp);
	}

	/*
	 * Now do the same with skipping.
	 */
	isc_lfsr_init(&lfsr1, 0, 32, 0x80000057U, 0, NULL, NULL);
	for (i = 0; i < 32; i++) {
		isc_lfsr_generate(&lfsr1, &state[i], 4);
		isc_lfsr_skip(&lfsr1, 32);
		printf("lfsr1:  state[%2d] = %08x\n", i, state[i]);
	}
	isc_lfsr_init(&lfsr1, 0, 32, 0x80000057U, 0, NULL, NULL);
	for (i = 0; i < 32; i++) {
		isc_lfsr_generate(&lfsr1, &temp, 4);
		isc_lfsr_skip(&lfsr1, 32);
		if (state[i] != temp)
			printf("lfsr1:  state[%2d] = %08x, "
			       "but new state is %08x\n",
			       i, state[i], temp);
	}

	/*
	 * Try to find the period of the LFSR.
	 *
	 *	x^16 + x^5 + x^3 + x^2 + 1
	 */
	isc_lfsr_init(&lfsr2, 0, 16, 0x00008016U, 0, NULL, NULL);
	for (i = 0; i < 32; i++) {
		isc_lfsr_generate(&lfsr2, &state[i], 4);
		printf("lfsr2:  state[%2d] = %08x\n", i, state[i]);
	}
	isc_lfsr_init(&lfsr2, 0, 16, 0x00008016U, 0, NULL, NULL);
	for (i = 0; i < 32; i++) {
		isc_lfsr_generate(&lfsr2, &temp, 4);
		if (state[i] != temp)
			printf("lfsr2:  state[%2d] = %08x, "
			       "but new state is %08x\n",
			       i, state[i], temp);
	}

	return (0);
}
