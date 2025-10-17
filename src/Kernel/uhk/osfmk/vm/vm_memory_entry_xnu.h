/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#ifndef _VM_VM_MEMORY_ENTRY_XNU_H_
#define _VM_VM_MEMORY_ENTRY_XNU_H_

#ifdef XNU_KERNEL_PRIVATE
#include <vm/vm_memory_entry.h>

__BEGIN_DECLS

extern void mach_memory_entry_port_release(ipc_port_t port);
extern vm_named_entry_t mach_memory_entry_from_port(ipc_port_t port);
extern struct vm_named_entry *mach_memory_entry_allocate(ipc_port_t *user_handle_p);

__END_DECLS
#endif /* XNU_KERNEL_PRIVATE */
#endif  /* _VM_VM_MEMORY_ENTRY_XNU_H_ */
