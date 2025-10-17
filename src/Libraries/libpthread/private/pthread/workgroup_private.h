/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 19, 2025.
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
#ifndef __PTHREAD_WORKGROUP_PRIVATE_H__
#define __PTHREAD_WORKGROUP_PRIVATE_H__

#include <pthread/pthread.h>
#include <os/workgroup.h>

__BEGIN_DECLS

/*!
 * @function pthread_create_with_workgroup_np
 *
 * @abstract
 * Creates a pthread that joins a specified workgroup and can never leave it.
 *
 * @param wg
 * The workgroup the new thread should join.  Must not be NULL.
 *
 * @result
 * Returns any result returned by pthread_create(3): zero if successful,
 * otherwise one of its documented error numbers.
 *
 * @discussion
 * Parameters follow pthread_create(3), with the addition of `wg`.
 *
 * To eventually terminate the thread, the `start_routine` must return - the
 * thread may not terminate using pthread_exit(3).
 *
 * Failure by the new thread to join the specified workgroup for any reason will
 * result in a crash, so clients must take care to ensure that the failures
 * documented for os_workgroup_join(3) won't occur.
 *
 * The thread may not leave its workgroup.
 */
SPI_AVAILABLE(macos(12.0), ios(15.0), tvos(15.0), watchos(8.0))
int
pthread_create_with_workgroup_np(pthread_t _Nullable * _Nonnull thread,
		os_workgroup_t _Nonnull wg, const pthread_attr_t * _Nullable attr,
		void * _Nullable (* _Nonnull start_routine)(void * _Nullable),
		void * _Nullable arg);

#if defined(PTHREAD_WORKGROUP_SPI) && PTHREAD_WORKGROUP_SPI

/*
 * Internal implementation details below.
 */

struct pthread_workgroup_functions_s {
#define PTHREAD_WORKGROUP_FUNCTIONS_VERSION 1
	int pwgf_version;
	// V1
	int (* _Nonnull pwgf_create_with_workgroup)(
			pthread_t _Nullable * _Nonnull thread, os_workgroup_t _Nonnull wg,
			const pthread_attr_t * _Nullable attr,
			void * _Nullable (* _Nonnull start_routine)(void * _Nullable),
			void * _Nullable arg);
};

SPI_AVAILABLE(macos(12.0), ios(15.0), tvos(15.0), watchos(8.0))
void
pthread_install_workgroup_functions_np(
		const struct pthread_workgroup_functions_s * _Nonnull pwgf);

#endif // defined(PTHREAD_WORKGROUP_SPI) && PTHREAD_WORKGROUP_SPI

__END_DECLS

#endif /* __PTHREAD_WORKGROUP_PRIVATE_H__ */
