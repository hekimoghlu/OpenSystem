/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
 * @OSF_COPYRIGHT@
 *
 */

#ifndef _VM_CPM_H_
#define _VM_CPM_H_

/*
 *	File:	vm/cpm_internal.h
 *	Author:	Alan Langerman
 *	Date:	April 1995 and January 1996
 *
 *	Contiguous physical memory allocator.
 */

#include <mach/mach_types.h>
#include <vm/vm_page.h>
/*
 *	Return a linked list of physically contiguous
 *	wired pages.  Caller is responsible for disposal
 *	via cpm_release.
 *
 *	These pages are all in "gobbled" state when
 *	wired is FALSE.
 */
extern kern_return_t
cpm_allocate(vm_size_t size, vm_page_t *list, ppnum_t max_pnum, ppnum_t pnum_mask, boolean_t wire, int flags);

#endif  /* _VM_CPM_H_ */
