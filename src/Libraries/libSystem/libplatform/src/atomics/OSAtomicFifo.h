/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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

//
//  OSAtomicFifo.h
//  libatomics
//
//  Created by Rokhini Prabhu on 4/7/20.
//

#ifndef _OS_ATOMIC_FIFO_QUEUE_
#define _OS_ATOMIC_FIFO_QUEUE_

#if defined(__arm64e__) && __has_feature(ptrauth_calls)
#include <ptrauth.h>

#define COMMPAGE_PFZ_BASE_AUTH_KEY ptrauth_key_process_independent_code
#define COMMPAGE_PFZ_FN_AUTH_KEY ptrauth_key_function_pointer
#define COMMPAGE_PFZ_BASE_DISCRIMINATOR ptrauth_string_discriminator("pfz")

#define COMMPAGE_PFZ_BASE_PTR __ptrauth(COMMPAGE_PFZ_BASE_AUTH_KEY, 1, COMMPAGE_PFZ_BASE_DISCRIMINATOR)

#define SIGN_PFZ_FUNCTION_PTR(ptr) ptrauth_sign_unauthenticated(ptr, COMMPAGE_PFZ_FN_AUTH_KEY, 0)

#else /* defined(__arm64e__) && __has_feature(ptrauth_calls) */

#define COMMPAGE_PFZ_BASE_AUTH_KEY 0
#define COMMPAGE_PFZ_FN_AUTH_KEY 0
#define COMMPAGE_PFZ_BASE_DISCRIMINATOR 0

#define COMMPAGE_PFZ_BASE_PTR

#define SIGN_PFZ_FUNCTION_PTR(ptr) ptr
#endif /* defined(__arm64e__) && __has_feature(ptrauth_calls) */

extern void *COMMPAGE_PFZ_BASE_PTR commpage_pfz_base;

#endif /* _OS_ATOMIC_FIFO_QUEUE_ */
