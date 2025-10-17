/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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
#ifndef PAS_EPOCH_H
#define PAS_EPOCH_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern bool pas_epoch_is_counter;
PAS_API extern uint64_t pas_current_epoch;

#define PAS_EPOCH_INVALID 0
#define PAS_EPOCH_MIN 1
#define PAS_EPOCH_MAX UINT64_MAX

/* This *may* simply return a new epoch each time you call it. Or it may return some coarser
   notion of epoch. The only requirement is that it proceeds monotonically, and even that
   requirement is a weak one - slight time travel is to be tolerated by callers.

   However: in reality this is monotonic time in nanoseconds. Lots of things are tuned for that
   fact.

   It's just that for testing purposes, we sometimes turn it into a counter and change the tuning
   accordingly. But that won't happen except in the tests that need it. */
PAS_API uint64_t pas_get_epoch(void);

PAS_END_EXTERN_C;

#endif /* PAS_EPOCH_H */

