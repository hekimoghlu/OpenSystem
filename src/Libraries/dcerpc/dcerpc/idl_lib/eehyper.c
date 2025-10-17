/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
**      eehyper.c
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**      Callee marshalling and unmarshalling of pointed_at hyper integers
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

void

rpc_ss_me_hyper_int
(
    idl_hyper_int *p_node,
    rpc_ss_node_type_k_t NIDL_node_type,
    rpc_ss_marsh_state_t *NIDL_msp
)
{
  long NIDL_already_marshalled;
  unsigned long space_for_node;
  rpc_mp_t mp;
  rpc_op_t op;

  if(p_node==NULL)return;
  if (NIDL_node_type == rpc_ss_mutable_node_k) {
      rpc_ss_register_node(NIDL_msp->node_table,(byte_p_t)p_node,idl_true,&NIDL_already_marshalled);
      if(NIDL_already_marshalled)return;
  }
  space_for_node=((8))+7;
  if (space_for_node > NIDL_msp->space_in_buff)
  {
    rpc_ss_marsh_change_buff(NIDL_msp,space_for_node);
  }
  mp = NIDL_msp->mp;
  op = NIDL_msp->op;
  rpc_align_mop(mp, op, 8);
  rpc_marshall_hyper_int(mp, (p_node));
  rpc_advance_mop(mp, op, 8);
  NIDL_msp->space_in_buff -= (op - NIDL_msp->op);
  NIDL_msp->mp = mp;
  NIDL_msp->op = op;
}

void

rpc_ss_ue_hyper_int
(
    idl_hyper_int **p_referred_to_by,
    rpc_ss_node_type_k_t NIDL_node_type,
    rpc_ss_marsh_state_t *p_unmar_params
)
{
  idl_hyper_int  *p_node = NULL;
  long NIDL_already_unmarshalled = 0;
  unsigned long node_size;
  idl_ulong_int node_number = 0;

  if ( NIDL_node_type == rpc_ss_unique_node_k )
  {
    if (*p_referred_to_by == NULL) return;
    else if (*p_referred_to_by != (idl_hyper_int *)RPC_SS_NEW_UNIQUE_NODE) p_node = *p_referred_to_by;
  }

  if ( NIDL_node_type == rpc_ss_mutable_node_k )
  {
    node_number = (idl_ulong_int)*p_referred_to_by;
    if(node_number==0)return;
  }
  if ( NIDL_node_type == rpc_ss_old_ref_node_k )
   p_node = *p_referred_to_by;
  else if ( p_node == NULL )
  {
    node_size = sizeof(idl_hyper_int );
    if (NIDL_node_type == rpc_ss_mutable_node_k)
    {
        p_node = (idl_hyper_int *) (void *) rpc_ss_return_pointer_to_node(
                p_unmar_params->node_table, node_number, node_size,
                NULL, &NIDL_already_unmarshalled, (long *)NULL);
    }
    else
    p_node = (idl_hyper_int *)rpc_ss_mem_alloc(
        p_unmar_params->p_mem_h, node_size );
    *p_referred_to_by = p_node;
    if (NIDL_already_unmarshalled) return;
  }
  if ( NIDL_node_type == rpc_ss_alloc_ref_node_k )
  {
    return;
  }
  rpc_align_mop(p_unmar_params->mp, p_unmar_params->op, 8);
  if ((unsigned32)((byte_p_t)p_unmar_params->mp - p_unmar_params->p_rcvd_data->data_addr) >= p_unmar_params->p_rcvd_data->data_len)
  {
    rpc_ss_new_recv_buff(p_unmar_params->p_rcvd_data, p_unmar_params->call_h, &(p_unmar_params->mp), &(*p_unmar_params->p_st));
  }
  rpc_convert_hyper_int(p_unmar_params->src_drep, ndr_g_local_drep, p_unmar_params->mp, (p_node));
  rpc_advance_mop(p_unmar_params->mp, p_unmar_params->op, 8);
}
