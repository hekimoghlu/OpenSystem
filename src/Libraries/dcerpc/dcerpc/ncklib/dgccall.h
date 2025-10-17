/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
**      dgccall.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG protocol service routines
**
**
*/

#ifndef _DGCCALL_H
#define _DGCCALL_H	1

#include <dgccallt.h>

/*
 * R P C _ D G _ C C A L L _ S E T _ S T A T E _ I D L E
 *
 * Remove the call handle from the CCALLT.  Release our reference to
 * our CCTE.  (In the case of CCALLs created to do server callbacks there
 * won't be a ccte_ref.)  Change to the idle state.  If you're trying
 * to get rid of the ccall use rpc__dg_ccall_free_prep() instead.
 */

#define RPC_DG_CCALL_SET_STATE_IDLE(ccall) { \
    if ((ccall)->c.state == rpc_e_dg_cs_final) \
        rpc__dg_ccall_ack(ccall); \
    rpc__dg_ccallt_remove(ccall); \
    if (! (ccall)->c.is_cbk)\
        RPC_DG_CCT_RELEASE(&(ccall)->ccte_ref); \
    RPC_DG_CALL_SET_STATE(&(ccall)->c, rpc_e_dg_cs_idle); \
}

/*
 * R P C _ D G _ C C A L L _ R E L E A S E
 *
 * Decrement the reference count for the CCALL and
 * NULL the reference.
 */

#define RPC_DG_CCALL_RELEASE(ccall) { \
    RPC_DG_CALL_LOCK_ASSERT(&(*(ccall))->c); \
    assert((*(ccall))->c.refcnt > 0); \
    if (--(*(ccall))->c.refcnt == 0) \
        rpc__dg_ccall_free(*(ccall)); \
    else \
        RPC_DG_CALL_UNLOCK(&(*(ccall))->c); \
    *(ccall) = NULL; \
}

/*
 * R P C _ D G _ C C A L L _ R E L E A S E _ N O _ U N L O C K
 *
 * Like RPC_DG_CCALL_RELEASE, except doesn't unlock the CCALL.  Note
 * that the referencing counting model requires that this macro can be
 * used iff the release will not be the "last one" (i.e., the one that
 * would normally cause the CCALL to be freed).
 */

#define RPC_DG_CCALL_RELEASE_NO_UNLOCK(ccall) { \
    RPC_DG_CALL_LOCK_ASSERT(&(*(ccall))->c); \
    assert((*(ccall))->c.refcnt > 1); \
    --(*(ccall))->c.refcnt; \
    *(ccall) = NULL; \
}

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__dg_ccall_lsct_inq_scall (
        rpc_dg_ccall_p_t  /*ccall*/,
        rpc_dg_scall_p_t * /*scallp*/
    );

PRIVATE void rpc__dg_ccall_lsct_new_call (
        rpc_dg_ccall_p_t  /*ccall*/,
        rpc_dg_sock_pool_elt_p_t  /*si*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_scall_p_t * /*scallp*/
    );

PRIVATE void rpc__dg_ccall_ack (
        rpc_dg_ccall_p_t /*ccall*/
    );

PRIVATE void rpc__dg_ccall_free (
        rpc_dg_ccall_p_t /*ccall*/
    );

PRIVATE void rpc__dg_ccall_free_prep (
        rpc_dg_ccall_p_t /*ccall*/
    );

PRIVATE void rpc__dg_ccall_timer ( dce_pointer_t /*p*/ );

PRIVATE void rpc__dg_ccall_xmit_cancel_quit (
        rpc_dg_ccall_p_t  /*ccall*/,
        unsigned32 /*cancel_id*/
    );

PRIVATE void rpc__dg_ccall_setup_cancel_tmo (
        rpc_dg_ccall_p_t /*ccall*/
    );

#ifdef __cplusplus
}
#endif

#endif /* _DGCCALL_H */
