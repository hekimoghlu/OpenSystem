/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
**  NAME:
**
**      pipesupp.c
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      Type independent routines to support pipes
**
**
*/
#if HAVE_CONFIG_H
#include <config.h>
#endif

/* The ordering of the following 3 includes should NOT be changed! */
#include <dce/rpc.h>
#include <dce/stubbase.h>
#include <lsysdep.h>

#ifdef PERFMON
#include <dce/idl_log.h>
#endif

#ifdef MIA
#include <dce/idlddefs.h>
#endif

#if 0
/* Removed unused symbol for rdar://problem/26430747 */
void rpc_ss_initialize_callee_pipe
(
    long pipe_index,    /* Index of pipe in set of pipes in the
                            operation's parameter list */
    long next_in_pipe,     /* Index of next [in] pipe to process */
    long next_out_pipe,     /* Index of next [out] pipe to process */
    long *p_current_pipe,    /* Ptr to index num and dirn of curr active pipe */
    rpc_mp_t *p_mp,         /* Ptr to marshalling pointer */
    rpc_op_t *p_op,     /* Ptr to offset pointer */
    ndr_format_t src_drep,   /* Sender's data representation */
    rpc_iovector_elt_t *p_rcvd_data, /* Addr of received data descriptor */
    rpc_ss_mem_handle *p_mem_h,    /* Ptr to stub memory allocation handle */
    rpc_call_handle_t call_h,
    rpc_ss_ee_pipe_state_t **p_p_pipe_state,    /* Addr of ptr to pipe state block */
    error_status_t *st
)
{
    rpc_ss_ee_pipe_state_t *p_pipe_state;

#ifdef PERFMON
    RPC_SS_INITIALIZE_CALLEE_PIPE_N;
#endif

    p_pipe_state = (rpc_ss_ee_pipe_state_t *)rpc_ss_mem_alloc(
                                    p_mem_h, sizeof(rpc_ss_ee_pipe_state_t));
    if (p_pipe_state == NULL)
    {
        DCETHREAD_RAISE(rpc_x_no_memory);
        return;
    }
    p_pipe_state->pipe_drained = ndr_false;
    p_pipe_state->pipe_filled = ndr_false;
    p_pipe_state->pipe_number = pipe_index;
    p_pipe_state->next_in_pipe = next_in_pipe;
    p_pipe_state->next_out_pipe = next_out_pipe;
    p_pipe_state->p_current_pipe = p_current_pipe;
    p_pipe_state->left_in_wire_array = 0;
    p_pipe_state->p_mp = p_mp;
    p_pipe_state->p_op = p_op;
    p_pipe_state->src_drep = src_drep;
    p_pipe_state->p_rcvd_data = p_rcvd_data;
    p_pipe_state->p_mem_h = p_mem_h;
    p_pipe_state->call_h = call_h;
    p_pipe_state->p_st = st;
    *p_p_pipe_state = p_pipe_state;
    *st = error_status_ok;

#ifdef PERFMON
    RPC_SS_INITIALIZE_CALLEE_PIPE_X;
#endif

}
#endif

#ifdef MIA

void rpc_ss_mts_init_callee_pipe
(
    long pipe_index,    /* Index of pipe in set of pipes in the
                            operation's parameter list */
    long next_in_pipe,     /* Index of next [in] pipe to process */
    long next_out_pipe,     /* Index of next [out] pipe to process */
    long *p_current_pipe,    /* Ptr to index num and dirn of curr active pipe */
    struct IDL_ms_t *IDL_msp,       /* Pointer to interpreter state block */
    unsigned long IDL_base_type_offset,  /* Offset of pipe base type definition
                                            in type vector */
    rpc_ss_mts_ee_pipe_state_t **p_p_pipe_state
                                           /* Addr of ptr to pipe state block */
)
{
    rpc_ss_mts_ee_pipe_state_t *p_pipe_state;

#ifdef PERFMON
    RPC_SS_INITIALIZE_CALLEE_PIPE_N;
#endif

    p_pipe_state = (rpc_ss_mts_ee_pipe_state_t *)
                                rpc_ss_mem_alloc(&IDL_msp->IDL_mem_handle,
                                sizeof(rpc_ss_mts_ee_pipe_state_t));
    if (p_pipe_state == NULL)
    {
        DCETHREAD_RAISE(rpc_x_no_memory);
        return;
    }
    p_pipe_state->pipe_drained = ndr_false;
    p_pipe_state->pipe_filled = ndr_false;
    p_pipe_state->pipe_number = pipe_index;
    p_pipe_state->next_in_pipe = next_in_pipe;
    p_pipe_state->next_out_pipe = next_out_pipe;
    p_pipe_state->p_current_pipe = p_current_pipe;
    p_pipe_state->left_in_wire_array = 0;
    p_pipe_state->IDL_msp = IDL_msp;
    p_pipe_state->IDL_base_type_offset = IDL_base_type_offset;
    *p_p_pipe_state = p_pipe_state;

#ifdef PERFMON
    RPC_SS_INITIALIZE_CALLEE_PIPE_X;
#endif

}
#endif
