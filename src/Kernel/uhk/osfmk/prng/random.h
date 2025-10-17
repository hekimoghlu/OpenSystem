/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#ifndef _PRNG_RANDOM_H_
#define _PRNG_RANDOM_H_

__BEGIN_DECLS

#include <corecrypto/cckprng.h>

#ifdef XNU_KERNEL_PRIVATE

void random_cpu_init(int cpu);

#endif /* XNU_KERNEL_PRIVATE */

void register_and_init_prng(struct cckprng_ctx *ctx, const struct cckprng_funcs *funcs);

#include <kern/simple_lock.h>
/* Definitions for boolean PRNG */
#define RANDOM_BOOL_GEN_SEED_COUNT 4
struct bool_gen {
	unsigned int seed[RANDOM_BOOL_GEN_SEED_COUNT];
	unsigned int state;
	decl_simple_lock_data(, lock);
};

extern void random_bool_init(struct bool_gen * bg);

extern void random_bool_gen_entropy(struct bool_gen * bg, unsigned int * buffer, int count);

extern unsigned int random_bool_gen_bits(struct bool_gen * bg, unsigned int * buffer, unsigned int count, unsigned int numbits);

__END_DECLS

#endif /* _PRNG_RANDOM_H_ */
