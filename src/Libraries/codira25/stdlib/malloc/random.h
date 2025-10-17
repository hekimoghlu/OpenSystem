/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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

#ifndef RANDOM_H
#define RANDOM_H

#include "chacha.h"
#include "util.h"

#define RANDOM_CACHE_SIZE 256U
#define RANDOM_RESEED_SIZE (256U * 1024)

struct random_state {
    unsigned index;
    unsigned reseed;
    chacha_ctx ctx;
    u8 cache[RANDOM_CACHE_SIZE];
};

void random_state_init(struct random_state *state);
void random_state_init_from_random_state(struct random_state *state, struct random_state *source);
void get_random_bytes(struct random_state *state, void *buf, size_t size);
u16 get_random_u16(struct random_state *state);
u16 get_random_u16_uniform(struct random_state *state, u16 bound);
u64 get_random_u64(struct random_state *state);
u64 get_random_u64_uniform(struct random_state *state, u64 bound);

#endif
