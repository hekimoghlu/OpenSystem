/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
/*
 * These routines are DEPRECATED and should not be used.
 */
#ifndef _UCONTEXT_H_
#define _UCONTEXT_H_

#include <sys/cdefs.h>

#ifdef _XOPEN_SOURCE
#include <sys/ucontext.h>
#include <Availability.h>

__BEGIN_DECLS
__API_DEPRECATED("No longer supported", macos(10.5, 10.6))
int  getcontext(ucontext_t *);

__API_DEPRECATED("No longer supported", macos(10.5, 10.6))
void makecontext(ucontext_t *, void (*)(), int, ...);

__API_DEPRECATED("No longer supported", macos(10.5, 10.6))
int  setcontext(const ucontext_t *);

__API_DEPRECATED("No longer supported", macos(10.5, 10.6))
int  swapcontext(ucontext_t * __restrict, const ucontext_t * __restrict);

__END_DECLS
#else /* !_XOPEN_SOURCE */
#error The deprecated ucontext routines require _XOPEN_SOURCE to be defined
#endif /* _XOPEN_SOURCE */

#endif /* _UCONTEXT_H_ */
