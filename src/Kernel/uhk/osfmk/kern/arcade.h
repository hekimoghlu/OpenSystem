/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#ifndef _KERN_ARCADE_H_
#define _KERN_ARCADE_H_

#include <mach/mach_types.h>
#include <kern/kern_types.h>

#include <libkern/section_keywords.h>


#if XNU_KERNEL_PRIVATE

struct arcade_register;

extern void arcade_init(void);

extern void arcade_ast(thread_t thread);

extern void arcade_prepare(task_t task, thread_t thread);

extern void arcade_register_notify(mach_msg_header_t *msg);

extern void arcade_register_reference(arcade_register_t arcade_reg);

extern void arcade_register_release(arcade_register_t arcade_reg);

extern mach_port_t convert_arcade_register_to_port(arcade_register_t arcade_reg);

extern arcade_register_t convert_port_to_arcade_register(mach_port_t port);

#endif /* XNU_KERNEL_PRIVATE */

#endif /* _KERN_ARCADE_H_ */
