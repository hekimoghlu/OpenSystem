/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
**      nidlalfr.c
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      rpc_ss_allocate, rpc_ss_free and helper thread routines
**
**  VERSION: DCE 1.0
**
*/
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <dce/rpc.h>
#include <dce/stubbase.h>
#include <lsysdep.h>

#ifdef PERFMON
#include <dce/idl_log.h>
#endif

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_allocate                                                         */
/*                                                                            */
/******************************************************************************/
idl_void_p_t rpc_ss_allocate
(
    idl_size_t size
)
{
    rpc_ss_thread_support_ptrs_t *p_support_ptrs = NULL;
    rpc_void_p_t                 p_new_node = NULL;
    error_status_t               status = rpc_s_ok;

#ifdef PERFMON
    RPC_SS_ALLOCATE_N;
#endif

    rpc_ss_get_support_ptrs( &p_support_ptrs );
    RPC_SS_THREADS_MUTEX_LOCK(&(p_support_ptrs->mutex));
    p_new_node = (rpc_void_p_t)rpc_sm_mem_alloc( p_support_ptrs->p_mem_h, size, &status );
    RPC_SS_THREADS_MUTEX_UNLOCK(&(p_support_ptrs->mutex));

    if (status == rpc_s_no_memory) DCETHREAD_RAISE( rpc_x_no_memory );

#ifdef PERFMON
    RPC_SS_ALLOCATE_X;
#endif

    return(p_new_node);

}

/******************************************************************************/
/*                                                                            */
/*    rpc_ss_free                                                             */
/*                                                                            */
/******************************************************************************/
void rpc_ss_free
(
    idl_void_p_t node_to_free
)
{
    rpc_ss_thread_support_ptrs_t *p_support_ptrs = NULL;

#ifdef PERFMON
    RPC_SS_FREE_N;
#endif

    rpc_ss_get_support_ptrs( &p_support_ptrs );
    RPC_SS_THREADS_MUTEX_LOCK(&(p_support_ptrs->mutex));
    if (p_support_ptrs->p_mem_h->node_table)
        /*
         * Must unregister node or a subsequent alloc could get same addr and
         * nodetbl mgmt would think it was an alias to storage's former life.
         */
        rpc_ss_unregister_node(p_support_ptrs->p_mem_h->node_table,
                               (byte_p_t)node_to_free);
    rpc_ss_mem_release(p_support_ptrs->p_mem_h, (byte_p_t)node_to_free, ndr_true);
    RPC_SS_THREADS_MUTEX_UNLOCK(&(p_support_ptrs->mutex));

#ifdef PERFMON
    RPC_SS_FREE_X;
#endif

}
