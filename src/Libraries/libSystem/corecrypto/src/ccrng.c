/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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

#include <corecrypto/ccrng.h>
#include <stdio.h>
#include <corecrypto/ccrng_system.h>

struct ccrng_system_state ccrng_global_system_rng_instance = {0};

struct ccrng_state* ccrng(int* error) {
	// we're basically using `fd` as a boolean for whether it's initialized or not
	// i mean, it's not like our implementation of the system RNG is using it Â¯\_(ãƒ„)_/Â¯
	if (ccrng_global_system_rng_instance.fd == 0) {
		if (ccrng_system_init(&ccrng_global_system_rng_instance)) {
			if (error)
				*error = 1;
			return NULL;
		}
		ccrng_global_system_rng_instance.fd = 1;
	}

	if (error)
		*error = 0;

	return (struct ccrng_state*)&ccrng_global_system_rng_instance;
};

int ccrng_uniform(struct ccrng_state *rng, uint64_t bound, uint64_t *rand) {
	// TODO(@facekapow): make this a proper uniform RNG
	// (actually, the whole RNG system needs to be fixed)
	//
	// the current implementation for this function does at least satisfy the requirement that the value be
	// between 0 and the upper bound, but i wouldn't say the number it generates has been "uniformly generated"

	uint64_t tmp = 0;
	ccrng_generate(rng, sizeof(tmp), &tmp);
	*rand = tmp % bound;
	return 0;
};
