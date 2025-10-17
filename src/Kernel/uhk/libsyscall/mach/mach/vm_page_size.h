/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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
#ifndef _VM_PAGE_SIZE_H_
#define _VM_PAGE_SIZE_H_

#include <Availability.h>
#include <mach/mach_types.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/*
 *	Globally interesting numbers.
 *	These macros assume vm_page_size is a power-of-2.
 */
extern  vm_size_t       vm_page_size;
extern  vm_size_t       vm_page_mask;
extern  int             vm_page_shift;

/*
 *	These macros assume vm_page_size is a power-of-2.
 */
#define trunc_page(x)   ((x) & (~(vm_page_size - 1)))
#define round_page(x)   trunc_page((x) + (vm_page_size - 1))

/*
 *	Page-size rounding macros for the fixed-width VM types.
 */
#define mach_vm_trunc_page(x) ((mach_vm_offset_t)(x) & ~((signed)vm_page_mask))
#define mach_vm_round_page(x) (((mach_vm_offset_t)(x) + vm_page_mask) & ~((signed)vm_page_mask))


extern  vm_size_t       vm_kernel_page_size     __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);
extern  vm_size_t       vm_kernel_page_mask     __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);
extern  int             vm_kernel_page_shift    __OSX_AVAILABLE_STARTING(__MAC_10_9, __IPHONE_7_0);

#define trunc_page_kernel(x)   ((x) & (~vm_kernel_page_mask))
#define round_page_kernel(x)   trunc_page_kernel((x) + vm_kernel_page_mask)

__END_DECLS

#endif
