/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#ifndef _CORE_EXCLUDE_H_
#define _CORE_EXCLUDE_H_

#include <mach/kern_return.h>
#include <mach/vm_types.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

#if KERNEL_PRIVATE

/*
 * Excludes a given memory region from the kernel coredump. This is recommended
 * for any sensitive data or data which is not expected to be necessary for
 * debugging.
 *
 * The address and size of the region must be page-aligned, the size must be
 * non-zero, and the addition of the address and size must not overflow.
 *
 * Note that you may need to call this function multiple times if the
 * underlying memory is shared.
 */
void
kdp_core_exclude_region(vm_offset_t addr, vm_size_t size);

/*
 * Unexcludes a given memory region from the kernel coredump.
 *
 * The address and size of the region must match a currently excluded region.
 */
void
kdp_core_unexclude_region(vm_offset_t addr, vm_size_t size);

#endif /* KERNEL_PRIVATE */

__END_DECLS

#endif /* _CORE_EXCLUDE_H_ */
