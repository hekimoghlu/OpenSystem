/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
**      marbfman.c
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      Buffer management when marshalling a node
**
**  VERSION: DCE 1.0
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

void rpc_ss_marsh_change_buff
(
    rpc_ss_marsh_state_t    *msp,  /* Pointer to marshalling state block */
    unsigned long size_next_structure
                            /* Size of next structure to be marshalled */
)
{
    ndr_byte *wp_buff;
    unsigned long req_buff_size;
    int preserved_offset;    /* Must start marshalling in new buffer at
                                same offset (mod 8) we were at in old buffer */

#ifdef PERFMON
    RPC_SS_MARSH_CHANGE_BUFF_N;
#endif

    preserved_offset = ((intptr_t)msp->mp) % 8;
    /* If a current iovector and buffer exist */
    if (msp->p_iovec->elt[0].buff_addr != NULL)
    {
        /* Despatch buffer to the comm layer */
        msp->p_iovec->elt[0].data_len = (unsigned32) (msp->p_iovec->elt[0].buff_len
              - (msp->p_iovec->elt[0].data_addr - msp->p_iovec->elt[0].buff_addr)
                                    - msp->space_in_buff);
        rpc_call_transmit ( msp->call_h, msp->p_iovec, msp->p_st );

#ifdef NO_EXCEPTIONS
        if (*msp->p_st != error_status_ok) return;
#else
        /* If cancelled, raise the cancelled exception */
        if (*msp->p_st==rpc_s_call_cancelled) DCETHREAD_RAISE(RPC_SS_THREADS_X_CANCELLED);

        /*
         *  Otherwise, raise the pipe comm error which causes the stub to
         *  report the value of the status variable.
         */
        if (*msp->p_st != error_status_ok) DCETHREAD_RAISE(rpc_x_ss_pipe_comm_error);
#endif
    }
    /* Get a new buffer */
    req_buff_size = size_next_structure + 18;
    /* 18 = 7 bytes worst case alignment before data, then we expect another
                node to follow, so
            3 bytes worst case alignment
            4 bytes offset
            4 bytes node number */
    if (req_buff_size < NIDL_BUFF_SIZE) req_buff_size = NIDL_BUFF_SIZE;
    req_buff_size += 7; /* malloc may not deliver 8-byte aligned memory */
    wp_buff = (ndr_byte *)malloc( req_buff_size );
    if (wp_buff == NULL)
    {
        DCETHREAD_RAISE( rpc_x_no_memory );
        return;
    }
    /* Fill in fields of the iovector */
    msp->p_iovec->num_elt = 1;
    msp->p_iovec->elt[0].buff_dealloc = (rpc_ss_dealloc_t)free;
    msp->p_iovec->elt[0].flags = 0;
    msp->p_iovec->elt[0].buff_addr = wp_buff;
    msp->p_iovec->elt[0].buff_len = (unsigned32) req_buff_size;
    /* malloc may not deliver 8-byte aligned memory */
    wp_buff = (byte_p_t)(((wp_buff - (byte_p_t)0) + 7) & ~7);
    msp->p_iovec->elt[0].data_addr = wp_buff + preserved_offset;
    /* Output parameters */
    msp->mp = (rpc_mp_t)msp->p_iovec->elt[0].data_addr;
    
    msp->space_in_buff = (unsigned32) (req_buff_size -
                  (msp->p_iovec->elt[0].data_addr - msp->p_iovec->elt[0].buff_addr));

#ifdef PERFMON
    RPC_SS_MARSH_CHANGE_BUFF_X;
#endif

}

