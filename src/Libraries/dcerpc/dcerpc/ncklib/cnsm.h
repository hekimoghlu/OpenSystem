/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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
**  NAME
**
**      cnsm.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Interface to the NCA Connection Protocol State Machine Service
**
**
*/

#ifndef _CNSM_H
#define _CNSM_H	1

/*
 * R P C _ _ C N _ S M _ I N I T
 */

void rpc__cn_sm_init (
    rpc_cn_sm_state_entry_p_t   */* state_tbl */,
    rpc_cn_sm_action_fn_p_t     /* action_tbl */,
    rpc_cn_sm_ctlblk_p_t         /* sm */,
    unsigned32			 /* tbl_id */);

/*
 * R P C _ _ C N _ S M _ E V A L _ E V E N T
 */

unsigned32     rpc__cn_sm_eval_event (
    unsigned32                  /* event_id */,
    dce_pointer_t                   /* event_parameter */,
    dce_pointer_t                   /* spc_struct */,
    rpc_cn_sm_ctlblk_p_t         /* sm */);

/*
 * R P C _ _ C N _ S M _ I N I T _ E V E N T _ L I S T
 */

void rpc__cn_sm_init_event_list (rpc_cn_sm_ctlblk_t  *);

/*
 * R P C _ _ C N _ S M _ I N S E R T _ E V E N T
 */

void rpc__cn_sm_insert_event (
    rpc_cn_sm_event_entry_p_t   /* event */,
    rpc_cn_sm_ctlblk_t          * /* sm */);

/*
**++
**
**  MACRO NAME:       RPC_CN_INCR_ACTIVE_SVR_ACTION
**
**  SCOPE:            Internal to the rpc;  used here and in cnassoc.c.
**
**  DESCRIPTION:
**
**  MACRO to set the active predicate to true. The server
**  runtime allocated the association for the new call and its
**  callbacks. Only one call and its related callbacks may allocate an
**  association at a time.  This macro includes the essence of
**  incr_active_action_rtn.
**
**  INPUTS:
**
**      assoc		Pointer to the association.
**
**   	sm 		The control block from the event evaluation
**                      routine.  Input is the current state and
**                      event for the control block.  Output is the
**                      next state or updated current state, for the
**                      control block.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            Modifies the association reference count and the
**			current state of the control block.
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
**/
#define RPC_CN_INCR_ACTIVE_SVR_ACTION(assoc, sm)\
{\
    	RPC_CN_DBG_RTN_PRINTF(SERVER rpc_cn_incr_active_svr_action_macro); \
	assoc->assoc_ref_count++;\
	sm->cur_state = RPC_C_SERVER_ASSOC_OPEN;\
}



/*
**++
**
**  MACRO NAME:         RPC_CN_INCR_ACTIVE_CL_ACTION
**
**  SCOPE:              GLOBAL
**
**  DESCRIPTION:
**
**  Action client side macro, to set the active predicate to true. The client
**  runtime allocated the association for the new call and its
**  callbacks. Only one call and its related callbacks may allocate an
**  association at a time.
**
**  INPUTS:
**
**      assoc		A pointer to the association.
**
**   	sm 		The control block from the event evaluation
**                      routine.  Input is the current state and
**                      event for the control block.  Output is the
**                      next state or updated current state, for the
**                      control block.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            Modifies the association reference count and the
**			current state of the control block.
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
*/
#define RPC_CN_INCR_ACTIVE_CL_ACTION(assoc, sm)\
{\
    RPC_CN_DBG_RTN_PRINTF(CLIENT rpc_cn_incr_active_cl_action_macro); \
    assoc->assoc_ref_count++;\
    sm->cur_state = RPC_C_CLIENT_ASSOC_ACTIVE;\
}
/*
**++
**
**  MACRO NAME:       RPC_CN_DECR_ACTIVE_SVR_ACTION
**
**  SCOPE:            Internal to the rpc;  used here and in cnassoc.c.
**
**  DESCRIPTION:
**
**  MACRO to set the active predicate to true. The server
**  runtime allocated the association for the new call and its
**  callbacks. Only one call and its related callbacks may allocate an
**  association at a time.  This macro includes the essence of
**  decr_active_action_rtn.
**
**  INPUTS:
**
**      assoc		Pointer to the association.
**
**   	sm 		The control block from the event evaluation
**                      routine.  Input is the current state and
**                      event for the control block.  Output is the
**                      next state or updated current state, for the
**                      control block.
**
**  INPUTS/OUTPUTS:     none
**
**  OUTPUTS:            Modifies the association reference count and the
**			current state of the control block.
**
**  IMPLICIT INPUTS:    none
**
**  IMPLICIT OUTPUTS:   none
**
**  FUNCTION VALUE:     none
**
**  SIDE EFFECTS:       none
**
**--
**/
#define RPC_CN_DECR_ACTIVE_SVR_ACTION(assoc, sm)\
{\
    	RPC_CN_DBG_RTN_PRINTF(SERVER rpc_cn_decr_active_svr_action_macro); \
	assoc->assoc_ref_count--;\
	sm->cur_state = RPC_C_SERVER_ASSOC_OPEN;\
}



#define RPC_CN_SM_GET_NEXT_EVENT(sm, event, more)\
{\
    if ((sm)->event_list_state == RPC_C_CN_SM_EVENT_LIST_EMPTY)\
    {\
        more = false;\
    }\
    else\
    {\
        (event)->event_id = (sm)->event_list[(sm)->event_list_hindex].event_id;\
        (event)->event_param = (sm)->event_list[(sm)->event_list_hindex].event_param;\
        (sm)->event_list_hindex = ((sm)->event_list_hindex + 1) &\
        (RPC_C_CN_SM_EVENT_LIST_MAX_ENTRIES - 1); \
        if ((sm)->event_list_hindex == (sm)->event_list_tindex)\
        {\
            (sm)->event_list_state = RPC_C_CN_SM_EVENT_LIST_EMPTY;\
        }\
        more = true;\
    }\
}

#endif
