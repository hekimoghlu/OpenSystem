/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
 *
 *	File: vm/vm_dyld_pager_internal.h
 *
 *      protos and definitions for dyld pager
 */

#ifndef _VM_DYLD_PAGER_INTERNAL_H_
#define _VM_DYLD_PAGER_INTERNAL_H_

#ifdef XNU_KERNEL_PRIVATE
#include <vm/vm_dyld_pager.h>

extern uint32_t dyld_pager_count;
extern uint32_t dyld_pager_count_max;

/*
 * VM call to implement map_with_linking_np() system call.
 */
extern kern_return_t
vm_map_with_linking(
	task_t                  task,
	struct mwl_region       *regions,
	uint32_t                region_cnt,
	void                    **link_info,
	uint32_t                link_info_size,
	memory_object_control_t file_control);

#endif /* KERNEL_PRIVATE */

#endif  /* _VM_DYLD_PAGER_INTERNAL_H_ */
