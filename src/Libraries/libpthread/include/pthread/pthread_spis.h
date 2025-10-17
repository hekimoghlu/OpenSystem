/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
 * Copyright 1996 1995 by Open Software Foundation, Inc. 1997 1996 1995 1994 1993 1992 1991
 *              All Rights Reserved
 *
 * Permission to use, copy, modify, and distribute this software and
 * its documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appears in all copies and
 * that both the copyright notice and this permission notice appear in
 * supporting documentation.
 *
 * OSF DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE.
 *
 * IN NO EVENT SHALL OSF BE LIABLE FOR ANY SPECIAL, INDIRECT, OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN ACTION OF CONTRACT,
 * NEGLIGENCE, OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 */
/*
 * MkLinux
 */

/*
 * Extension SPIs; installed to /usr/include.
 */

#ifndef _PTHREAD_SPIS_H
#define _PTHREAD_SPIS_H


#include <pthread/pthread.h>

#if __has_feature(assume_nonnull)
_Pragma("clang assume_nonnull begin")
#endif
__BEGIN_DECLS

#if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE)
/* firstfit */
#define PTHREAD_FIRSTFIT_MUTEX_INITIALIZER {_PTHREAD_FIRSTFIT_MUTEX_SIG_init, {0}}

/*
 * Mutex attributes
 */
#define _PTHREAD_MUTEX_POLICY_NONE			PTHREAD_MUTEX_POLICY_NONE
#define _PTHREAD_MUTEX_POLICY_FAIRSHARE		PTHREAD_MUTEX_POLICY_FAIRSHARE_NP
#define _PTHREAD_MUTEX_POLICY_FIRSTFIT		PTHREAD_MUTEX_POLICY_FIRSTFIT_NP

#endif /* (!_POSIX_C_SOURCE && !_XOPEN_SOURCE) || _DARWIN_C_SOURCE */

__API_AVAILABLE(macos(10.11)) __API_UNAVAILABLE(ios, tvos, watchos, bridgeos)
void _pthread_mutex_enable_legacy_mode(void);

/*
 * A version of pthread_create that is safely callable from an injected mach thread.
 *
 * The _create introspection hook will not fire for threads created from this function.
 *
 * It is not safe to call this function concurrently.
 */
__API_AVAILABLE(macos(10.12), ios(10.0), tvos(10.0), watchos(3.0))
#if !_PTHREAD_SWIFT_IMPORTER_NULLABILITY_COMPAT()
int pthread_create_from_mach_thread(
		pthread_t _Nullable * _Nonnull __restrict,
		const pthread_attr_t * _Nullable __restrict,
		void * _Nullable (* _Nonnull)(void * _Nullable),
		void * _Nullable __restrict);
#else
int pthread_create_from_mach_thread(pthread_t * __restrict,
		const pthread_attr_t * _Nullable __restrict,
		void *(* _Nonnull)(void *), void * _Nullable __restrict);
#endif


__END_DECLS
#if __has_feature(assume_nonnull)
_Pragma("clang assume_nonnull end")
#endif

#endif /* _PTHREAD_SPIS_H */
