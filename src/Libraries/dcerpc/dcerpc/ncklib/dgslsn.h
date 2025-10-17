/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
**      dgslsn.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG protocol service routines.  Server-oriented routines that run in the
**  listener thread.
**
**
*/

#ifndef _DGSLSN_H
#define _DGSLSN_H

PRIVATE boolean32 rpc__dg_svr_chk_and_set_sboot (
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_sock_pool_elt_p_t  /*sp*/
    );

PRIVATE boolean rpc__dg_do_quit (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_scall_p_t  /*scall*/
    );

PRIVATE boolean rpc__dg_do_ack (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_scall_p_t  /*scall*/
     );

PRIVATE boolean rpc__dg_do_ping (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_scall_p_t  /*scall*/
    );

PRIVATE boolean rpc__dg_do_request (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/
    );

#endif /* _DGSLSN_H */
