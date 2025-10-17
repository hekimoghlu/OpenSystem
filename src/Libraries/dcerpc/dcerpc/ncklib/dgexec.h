/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
**
**  NAME:
**
**      dgexec.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG protocol service routines.
**
**
*/

#ifndef _DGEXEC_H
#define _DGEXEC_H

PRIVATE void rpc__dg_execute_call    (
        dce_pointer_t  /*scall_*/,
        boolean32  /*call_was_queued*/
    );

/*
 * To implement backward compatibilty, declare a pointer to a routine
 * that can call into pre-v2 server stubs.  We declare this function
 * in this indirect way so that it is possible to build servers that
 * don't support backward compatibility (and thus save space).  The
 * compatibility code will only be linked into a server if the server
 * application code itself calls a compatibility function, most likely
 * rpc_$register.  rpc_$register is then responsible for initializing
 * this function pointer so that dg_execute_call can find the compatibilty
 * function.  In this way, libnck has no direct references to the
 * compatibilty code.
 */

typedef void (*rpc__dg_pre_v2_server_fn_t) (
        rpc_if_rep_p_t  /*ifspec*/,
        unsigned32  /*opnum*/,
        handle_t  /*h*/,
        rpc_call_handle_t  /*call_h*/,
        rpc_iovector_elt_p_t  /*iove_ins*/,
        ndr_format_t  /*ndr_format*/,
        rpc_v2_server_stub_epv_t  /*server_stub_epv*/,
        rpc_mgr_epv_t  /*mgr_epv*/,
        unsigned32 * /*st*/
    );

#endif /* _DGEXEC_H */
