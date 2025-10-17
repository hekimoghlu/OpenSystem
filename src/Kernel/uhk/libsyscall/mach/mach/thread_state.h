/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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
#ifndef _MACH_THREAD_STATE_H_
#define _MACH_THREAD_STATE_H_

#include <Availability.h>
#include <mach/mach.h>

#ifndef KERNEL
/*
 * Gets all register values in the target thread with pointer-like contents.
 *
 * There is no guarantee that the returned values are valid pointers, but all
 * valid pointers will be returned.  The order and count of the provided
 * register values is unspecified and may change; registers with values that
 * are not valid pointers may be omitted, so the number of pointers returned
 * may vary from call to call.
 *
 * sp is an out parameter that will contain the stack pointer.
 * length is an in/out parameter for the length of the values array.
 * values is an array of pointers.
 *
 * This may only be called on threads in the current task.  If the current
 * platform defines a stack red zone, the stack pointer returned will be
 * adjusted to account for red zone.
 *
 * If length is insufficient, KERN_INSUFFICIENT_BUFFER_SIZE will be returned
 * and length set to the amount of memory required.  Callers MUST NOT assume
 * that any particular size of buffer will be sufficient and should retry with
 * an appropriately sized buffer upon this error.
 */
__API_AVAILABLE(macosx(10.14), ios(12.0), tvos(9.0), watchos(5.0))
kern_return_t thread_get_register_pointer_values(thread_t thread,
    uintptr_t *sp, size_t *length, uintptr_t *values);
#endif

#endif /* _MACH_THREAD_STATE_H_ */
