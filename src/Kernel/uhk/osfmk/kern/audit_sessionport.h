/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#ifndef _KERN_AUDIT_SESSIONPORT_H_
#define _KERN_AUDIT_SESSIONPORT_H_

struct auditinfo_addr;

ipc_port_t audit_session_mksend(struct auditinfo_addr *, ipc_port_t *);
struct auditinfo_addr *audit_session_porttoaia(ipc_port_t);
void audit_session_portdestroy(ipc_port_t *);
void audit_session_aiaref(struct auditinfo_addr *);
void audit_session_aiaunref(struct auditinfo_addr *);

#endif /* _KERN_AUDIT_SESSIONPORT_H_ */
#endif /* KERNEL_PRIVATE */
