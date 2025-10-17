/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#ifndef _KERN_TASK_REF_H_
#define _KERN_TASK_REF_H_

#include <mach/mach_types.h>

#include <stdint.h>

#if MACH_KERNEL_PRIVATE


extern void task_ref_init(void);

extern void task_ref_count_fini(task_t);
extern kern_return_t task_ref_count_init(task_t);

extern void task_reference_external(task_t task);
extern void task_deallocate_external(task_t task);

#endif /* MACH_KERNEL_PRIVATE */

#if XNU_KERNEL_PRIVATE

#include <os/refcnt.h>

__BEGIN_DECLS

extern struct os_refgrp task_external_refgrp;

__options_closed_decl(task_grp_t, uint32_t, {
	TASK_GRP_KERNEL,
	TASK_GRP_INTERNAL,
	TASK_GRP_MIG,
	TASK_GRP_EXTERNAL,

	TASK_GRP_COUNT,
});

extern void task_reference_grp(task_t, task_grp_t);
extern void task_deallocate_grp(task_t, task_grp_t);

#define task_reference_mig(task) task_reference_grp(task, TASK_GRP_MIG)
#define task_deallocate_mig(task) task_deallocate_grp(task, TASK_GRP_MIG)

/*
 * Exported symbols get mapped to their _external versions. Internal consumers of
 * these functions need to pick up the _kernel version.
 */

#define task_reference(task) task_reference_grp(task, TASK_GRP_KERNEL)
#define task_deallocate(task) task_deallocate_grp(task, TASK_GRP_KERNEL)

#define convert_task_to_port(task) convert_task_to_port_kernel(task)
#define convert_task_read_to_port(task) convert_task_read_to_port_kernel(task)

#define port_name_to_task(name) port_name_to_task_kernel(name)

#define convert_port_to_task_suspension_token(port) convert_port_to_task_suspension_token_kernel(port)
#define convert_task_suspension_token_to_port(token) convert_task_suspension_token_to_port_kernel(token)

#define task_resume2(token) task_resume2_kernel(token)
#define task_suspend2(task, token) task_suspend2_kernel(task, token)

__END_DECLS

#else /* XNU_KERNEL_PRIVATE */

__BEGIN_DECLS

extern void             task_reference(task_t);
extern void             task_deallocate(task_t);

__END_DECLS
#endif /* XNU_KERNEL_PRIVATE */

#endif /*_KERN_TASK_REF_H_ */
