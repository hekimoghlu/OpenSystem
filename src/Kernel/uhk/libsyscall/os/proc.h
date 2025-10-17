/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
#ifndef __OS_PROC__
#define __OS_PROC__

#include <stddef.h>
#include <sys/cdefs.h>
#include <os/availability.h>

/*!
 * @header
 *
 * @preprocinfo
 * This is for functions that operate on the calling process alone.
 */

__BEGIN_DECLS

/*!
 * @function os_proc_available_memory
 *
 * @abstract
 * Return the number of bytes remaining, at the time of the call, before the
 * current process will hit its current dirty memory limit.
 *
 * @discussion
 * Developers can query this value efficiently whenever it is needed. The return
 * value is only a snapshot at the time of the call. Caching the result is not
 * advised. The result may be instantaneously invalidated by actions taken in
 * another thread or another framework.
 *
 * Memory limits can change during the app life cycle. Make sure to check accordingly.
 *
 * The size returned is not representative of the total memory of the device, it
 * is the current dirty memory limit minus the dirty memory footprint used at the
 * time of the query.
 *
 * This interface allows an app to efficiently consume all available memory resources.
 * Significant memory use, even under the current memory limit, may still cause
 * system-wide performance including the termination of other apps and system
 * processes. Take care to use the minimum amount of memory needed to satisfy the
 * userâ€™s need.
 *
 * If you need more information than just the available memory, you can use task_info().
 * The information returned is equivalent to the task_vm_info.limit_bytes_remaining
 * field. task_info() is a more expensive call, and will return information such
 * as your phys_footprint, which is used to calculate the return of this function.
 *
 * Dirty memory contains data that must be kept in RAM (or the equivalent) even
 * when unused. It is memory that has been modified.
 *
 * @param none
 *
 * @result
 * The remaining bytes. 0 is returned if the calling process is not an app, or
 * the calling process exceeds its memory limit.
 */

API_UNAVAILABLE(macos) API_AVAILABLE(ios(13.0), tvos(13.0), watchos(6.0), bridgeos(4.0))
extern
size_t os_proc_available_memory(void);

__END_DECLS

#endif
