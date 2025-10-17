/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#ifndef _VM_MEMORY_OBJECT_XNU_H_
#define _VM_MEMORY_OBJECT_XNU_H_

#ifdef XNU_KERNEL_PRIVATE

/* Also defined in vm_protos.h currently */
#if CONFIG_SECLUDED_MEMORY
extern void             memory_object_mark_eligible_for_secluded(
	memory_object_control_t         control,
	boolean_t                       eligible_for_secluded);
#endif /* CONFIG_SECLUDED_MEMORY */

extern void             memory_object_mark_for_realtime(
	memory_object_control_t         control,
	bool                            for_realtime);

#endif /* XNU_KERNEL_PRIVATE */

#endif  /* _VM_MEMORY_OBJECT_XNU_H_ */
