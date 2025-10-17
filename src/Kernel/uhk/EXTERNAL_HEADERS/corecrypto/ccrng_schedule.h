/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
#ifndef _CORECRYPTO_CCRNG_SCHEDULE_H_
#define _CORECRYPTO_CCRNG_SCHEDULE_H_

#include <corecrypto/cc.h>
#include <corecrypto/ccdrbg.h>
#include <stdatomic.h>

// Depending on the environment and platform APIs available, different
// RNGs will use different reseed strategies. For example, an RNG that
// can communicate with its entropy source might check a flag set by
// the latter when entropy is available. In another case, the RNG
// might poll its entropy source on a time interval. In some cases,
// the RNG might always (or never) want to try to reseed.
//
// This module provides a common interface for such reseed
// schedules. It is intended for use as a component by RNG
// implementations.

typedef enum {
    CCRNG_SCHEDULE_CONTINUE = 1,
    CCRNG_SCHEDULE_TRY_RESEED = 2,
    CCRNG_SCHEDULE_MUST_RESEED = 3,
} ccrng_schedule_action_t;

typedef struct ccrng_schedule_ctx ccrng_schedule_ctx_t;

// The schedule interface provides two function pointers: one to check
// the schedule and one to notify the schedule of a successful reseed.
typedef struct ccrng_schedule_info {
    ccrng_schedule_action_t (*read)(ccrng_schedule_ctx_t *ctx);
    void (*notify_reseed)(ccrng_schedule_ctx_t *ctx);
} ccrng_schedule_info_t;

struct ccrng_schedule_ctx {
    const ccrng_schedule_info_t *info;
    bool must_reseed;
};

ccrng_schedule_action_t ccrng_schedule_read(ccrng_schedule_ctx_t *ctx);

void ccrng_schedule_notify_reseed(ccrng_schedule_ctx_t *ctx);

// This is a concrete schedule implementation where the state of the
// entropy source is communicated via a flag. The entropy source can
// set the flag with ccrng_schedule_atomic_flag_set to indicate
// entropy is available. The flag is cleared automatically on read and
// reset if the reseed fails.
typedef struct ccrng_schedule_atomic_flag_ctx {
    ccrng_schedule_ctx_t schedule_ctx;
    _Atomic ccrng_schedule_action_t flag;
} ccrng_schedule_atomic_flag_ctx_t;

void ccrng_schedule_atomic_flag_init(ccrng_schedule_atomic_flag_ctx_t *ctx);

void ccrng_schedule_atomic_flag_set(ccrng_schedule_atomic_flag_ctx_t *ctx);

// This is a concrete schedule implementation that simply always
// returns a constant action.
typedef struct ccrng_schedule_constant_ctx {
    ccrng_schedule_ctx_t schedule_ctx;
    ccrng_schedule_action_t action;
} ccrng_schedule_constant_ctx_t;

void ccrng_schedule_constant_init(ccrng_schedule_constant_ctx_t *ctx,
                                  ccrng_schedule_action_t action);

// This is a concrete schedule implementation that returns "must
// reseed" over a given interval of time.
typedef struct ccrng_schedule_timer_ctx {
    ccrng_schedule_ctx_t schedule_ctx;
    uint64_t (*get_time)(void);
    uint64_t reseed_interval;
    uint64_t last_reseed_time;
} ccrng_schedule_timer_ctx_t;

void ccrng_schedule_timer_init(ccrng_schedule_timer_ctx_t *ctx,
                               uint64_t (*get_time)(void),
                               uint64_t reseed_interval);

// This is a concrete schedule implementation that combines the
// results of two constituent sub-schedules. Specifically, it returns
// the more "urgent" recommendation between the two.
typedef struct ccrng_schedule_tree_ctx {
    ccrng_schedule_ctx_t schedule_ctx;
    ccrng_schedule_ctx_t *left;
    ccrng_schedule_ctx_t *right;
} ccrng_schedule_tree_ctx_t;

void ccrng_schedule_tree_init(ccrng_schedule_tree_ctx_t *ctx,
                              ccrng_schedule_ctx_t *left,
                              ccrng_schedule_ctx_t *right);

typedef struct ccrng_schedule_drbg_ctx {
    ccrng_schedule_ctx_t schedule_ctx;
    const struct ccdrbg_info *drbg_info;
    struct ccdrbg_state *drbg_ctx;
} ccrng_schedule_drbg_ctx_t;

void ccrng_schedule_drbg_init(ccrng_schedule_drbg_ctx_t *ctx,
                              const struct ccdrbg_info *drbg_info,
                              struct ccdrbg_state *drbg_ctx);

#endif /* _CORECRYPTO_CCRNG_SCHEDULE_H_ */
