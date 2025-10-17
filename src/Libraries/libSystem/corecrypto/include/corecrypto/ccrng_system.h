/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
#ifndef _CORECRYPTO_CCRNG_SYSTEM_H_
#define _CORECRYPTO_CCRNG_SYSTEM_H_

#include <corecrypto/ccrng.h>

struct ccrng_system_state {
    CCRNG_STATE_COMMON
    int fd;
};

/*!
 @function   ccrng_system_init - DEPRECATED
 @abstract   Default ccrng.
    Please transition to ccrng() which is easier to use and with provide the fastest, most secure option

 @param  rng   Structure containing the state of the RNG, must remain allocated as
 long as the rng is used.
 @result 0 iff successful

 @discussion
        This RNG require call to "init" AND "done", otherwise it may leak a file descriptor.
 */

// Initialize ccrng
// Deprecated, if you need a rng, just call the function ccrng()
int ccrng_system_init(struct ccrng_system_state *rng);

// Close the system RNG
// Mandatory step to avoid leaking file descriptor
void ccrng_system_done(struct ccrng_system_state *rng);

#endif /* _CORECRYPTO_CCRNG_SYSTEM_H_ */
