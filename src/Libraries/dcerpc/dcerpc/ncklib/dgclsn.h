/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
**      dgclsn.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG protocol service routines.  Client-oriented routines that run in the
**  listener thread.
**
**
*/

#ifndef _DGCLSN_H
#define _DGCLSN_H	1

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE boolean rpc__dg_do_common_response (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_ccall_p_t  /*ccall*/
    );

PRIVATE boolean rpc__dg_do_reject (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_ccall_p_t  /*ccall*/
    );

PRIVATE boolean rpc__dg_do_fault (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_ccall_p_t  /*ccall*/
    );

PRIVATE boolean rpc__dg_do_response (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_ccall_p_t  /*ccall*/
    );

PRIVATE boolean rpc__dg_do_working (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_ccall_p_t  /*ccall*/
    );

PRIVATE boolean rpc__dg_do_nocall (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_ccall_p_t  /*ccall*/
    );

PRIVATE boolean rpc__dg_do_quack (
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_ccall_p_t  /*ccall*/
    );

#ifdef __cplusplus
}
#endif

#endif /* _DGCLSN_H */
