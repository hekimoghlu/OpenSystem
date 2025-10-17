/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#ifndef PAS_RANDOM_H
#define PAS_RANDOM_H

#include "pas_utils.h"
#include <stdlib.h>

PAS_BEGIN_EXTERN_C;

extern PAS_API unsigned pas_fast_random_state;

/* This is useful for testing. */
extern PAS_API unsigned (*pas_mock_fast_random)(void);

/* This is a PRNG optimized for speed and nothing else. It's used whenever we need a random number only as a
   performance optimization. Returns a random number in [0, upper_bound). If the upper_bound is set to zero, than
   the range shall be [0, UINT32_MAX). */
static inline unsigned pas_get_fast_random(unsigned upper_bound)
{
    unsigned rand_value;

    if (!upper_bound)
        upper_bound = UINT32_MAX;

    if (PAS_LIKELY(!pas_mock_fast_random)) {
        pas_fast_random_state = pas_xorshift32(pas_fast_random_state);
        rand_value = pas_fast_random_state % upper_bound;
    } else {
        /* This is testing code. It will not be called during regular code flow. */
        rand_value = pas_mock_fast_random() % upper_bound;
    }

    return rand_value;
}

/* This is a PRNG optimized for security. It's used whenever we need unpredictable data. This will incur significant
  performance penalties over pas_fast_random. Returns a random number in [0, upper_bound). If the upper_bound is set
  to zero, than the range shall be [0, UINT32_MAX). */
static inline unsigned pas_get_secure_random(unsigned upper_bound)
{
    unsigned rand_value;

    if (!upper_bound)
        upper_bound = UINT32_MAX;

    /* Secure random is only supported on Darwin and FreeBSD at the moment due to arc4random being built into the
      stdlib. Fall back to fast behavior on other operating systems. */
#if PAS_OS(DARWIN) || PAS_OS(FREEBSD)
    rand_value = arc4random_uniform(upper_bound);
#else
    pas_fast_random_state = pas_xorshift32(pas_fast_random_state);
    rand_value = pas_fast_random_state % upper_bound;
#endif

    return rand_value;
}

PAS_END_EXTERN_C;

#endif /* PAS_RANDOM_H */
