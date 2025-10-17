/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#ifdef KERNEL_PRIVATE
#ifndef _KERN_IPC_MISC_H_
#define _KERN_IPC_MISC_H_

struct fileglob;
ipc_port_t fileport_alloc(struct fileglob *);
struct fileglob *fileport_port_to_fileglob(ipc_port_t);
void fileport_notify(mach_msg_header_t *);
kern_return_t fileport_invoke(task_t, mach_port_name_t,
    int (*)(mach_port_name_t, struct fileglob *, void *), void *, int *);
kern_return_t fileport_walk(task_t, size_t *count,
    bool (^cb)(size_t i, mach_port_name_t, struct fileglob *));

#endif /* _KERN_IPC_MISC_H_ */
#endif /* KERNEL_PRIVATE */
