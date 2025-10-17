/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#ifndef _PTHREAD_SPAWN_H
#define _PTHREAD_SPAWN_H

/*!
 * @group posix_spawn QOS class support
 * Apple extensions to posix_spawn(2) and posix_spawnp(2)
 */

#include <pthread/pthread.h>
#include <spawn.h>

__BEGIN_DECLS

/*!
 * @function posix_spawnattr_set_qos_class_np
 *
 * @abstract
 * Sets the QOS class property of a posix_spawn attributes object, which may be
 * used to specify the QOS class a process should be spawned with.
 *
 * @discussion
 * The QOS class specified at the time of process spawn determines both the
 * initial requested QOS class of the main thread in the new process, and the
 * interpretation by the system of all QOS class values requested by threads in
 * the process.
 *
 * @param __attr
 * The spawn attributes object to modify.
 *
 * @param __qos_class
 * A QOS class value:
 *	- QOS_CLASS_UTILITY
 *	- QOS_CLASS_BACKGROUND
 * EINVAL will be returned if any other value is provided.
 *
 * @return
 * Zero if successful, otherwise an errno value.
 */
__API_AVAILABLE(macos(10.10), ios(8.0))
int
posix_spawnattr_set_qos_class_np(posix_spawnattr_t * __restrict __attr,
                                 qos_class_t __qos_class);

/*!
 * @function posix_spawnattr_get_qos_class_np
 *
 * @abstract
 * Gets the QOS class property of a posix_spawn attributes object.
 *
 * @param __attr
 * The spawn attributes object to inspect.
 *
 * @param __qos_class
 * On output, a QOS class value:
 *	- QOS_CLASS_UTILITY
 *	- QOS_CLASS_BACKGROUND
 *	- QOS_CLASS_UNSPECIFIED
 *
 * @return
 * Zero if successful, otherwise an errno value.
 */
__API_AVAILABLE(macos(10.10), ios(8.0))
int
posix_spawnattr_get_qos_class_np(const posix_spawnattr_t *__restrict __attr,
                                 qos_class_t * __restrict __qos_class);

__END_DECLS

#endif // _PTHREAD_SPAWN_H
