/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#ifndef __PTHREAD_DEPENDENCY_PRIVATE__
#define __PTHREAD_DEPENDENCY_PRIVATE__

#include <os/base.h>
#include <stdint.h>
#include <sys/cdefs.h>
#include <pthread/pthread.h>
#include <Availability.h>

__BEGIN_DECLS

OS_ASSUME_NONNULL_BEGIN

/*!
 * @typedef pthread_dependency_t
 *
 * @abstract
 * A pthread dependency is a one-time dependency between a thread producing
 * a value and a waiter thread, expressed to the system in a way
 * that priority inversion avoidance can be applied if necessary.
 *
 * @discussion
 * These tokens are one-time use, and meant to be on the stack of the waiter
 * thread.
 *
 * These tokens must be both fulfilled and waited on, exactly one of each.
 */
typedef struct pthread_dependency_s {
	uint32_t __pdep_owner;
	uint32_t __pdep_opaque1;
	uint64_t __pdep_opaque2;
} pthread_dependency_t;

/*!
 * @typedef pthread_dependency_attr_t
 *
 * @abstract
 * An opaque type to allow for future expansion of the pthread_dependency
 * interface.
 */
typedef struct pthread_dependency_attr_s pthread_dependency_attr_t;

#if (!defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)) || defined(_DARWIN_C_SOURCE) || defined(__cplusplus)
/*!
 * @macro PTHREAD_DEPENDENCY_INITIALIZER_NP
 *
 * @abstract
 * Initialize a one-time dependency token.
 *
 * @param __pthread
 * The thread that will be waited on for this dependency to be fulfilled.
 * It is expected that this thread will call pthread_dependency_fulfill_np().
 */
#define PTHREAD_DEPENDENCY_INITIALIZER_NP(__pthread) \
		{ pthread_mach_thread_np(__pthread), 0, 0 }
#endif

/*!
 * @function pthread_dependency_init_np
 *
 * @abstract
 * Initialize a dependency token.
 *
 * @param __dependency
 * A pointer to a dependency token to initialize.
 *
 * @param __pthread
 * The thread that will be waited on for this dependency to be fulfilled.
 * It is expected that this thread will call pthread_dependency_fulfill_np().
 *
 * @param __attrs
 * This argument is reserved for future expansion purposes, and NULL should be
 * passed.
 */
__API_AVAILABLE(macos(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_NONNULL1 OS_NONNULL2 OS_NOTHROW
void pthread_dependency_init_np(pthread_dependency_t *__dependency,
		pthread_t __pthread, pthread_dependency_attr_t *_Nullable __attrs);

/*!
 * @function pthread_dependency_fulfill_np
 *
 * @abstract
 * Fulfill a dependency.
 *
 * @discussion
 * Calling pthread_dependency_fulfill_np() with a token that hasn't been
 * initialized yet, or calling pthread_dependency_fulfill_np() on the same
 * dependency token more than once is undefined and will cause the process
 * to be terminated.
 *
 * The thread that calls pthread_dependency_fulfill_np() must be the same
 * as the pthread_t that was specified when initializing the token. Not doing so
 * is undefined and will cause the process to be terminated.
 *
 * @param __dependency
 * A pointer to a dependency token that was previously initialized.
 *
 * @param __value
 * An optional value that can be returned through the dependency token
 * to the waiter.
 */
__API_AVAILABLE(macos(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_NONNULL1 OS_NOTHROW
void pthread_dependency_fulfill_np(pthread_dependency_t *__dependency,
		void * _Nullable __value);

/*!
 * @function pthread_dependency_wait_np
 *
 * @abstract
 * Wait on a dependency.
 *
 * @discussion
 * Calling pthread_dependency_wait_np() with a token that hasn't been
 * initialized yet, or calling pthread_dependency_wait_np() on the same
 * dependency token more than once is undefined and will cause the process
 * to be terminated.
 *
 * If the dependency is not fulfilled yet when this function is called, priority
 * inversion avoidance will be applied to the thread that was specified when
 * initializing the token, to ensure that it can call
 * pthread_dependency_fulfill_np() without causing a priority inversion for the
 * thread calling pthread_dependency_wait_np().
 *
 * @param __dependency
 * A pointer to a dependency token that was previously initialized with
 * PTHREAD_DEPENDENCY_INITIALIZER_NP() or pthread_dependency_init_np().
 *
 * @returns
 * The value that was passed to pthread_dependency_fulfill_np() as the `__value`
 * argument.
 */
__API_AVAILABLE(macos(10.14), ios(12.0), tvos(12.0), watchos(5.0))
OS_NONNULL1 OS_NOTHROW
void *_Nullable pthread_dependency_wait_np(pthread_dependency_t *__dependency);

OS_ASSUME_NONNULL_END

__END_DECLS

#endif //__PTHREAD_DEPENDENCY_PRIVATE__
