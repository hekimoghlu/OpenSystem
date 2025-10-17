/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
**      dgscall.h
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

#ifndef _DGSCALL_H
#define _DGSCALL_H

/*
 * R P C _ D G _ S C A L L _ R E L E A S E
 *
 * Decrement the reference count for the SCALL and
 * NULL the reference.
 */

#define RPC_DG_SCALL_RELEASE(scall) { \
    RPC_DG_CALL_LOCK_ASSERT(&(*(scall))->c); \
    assert((*(scall))->c.refcnt > 0); \
    if (--(*(scall))->c.refcnt == 0) \
        rpc__dg_scall_free(*(scall)); \
    else \
        RPC_DG_CALL_UNLOCK(&(*(scall))->c); \
    *(scall) = NULL; \
}

/*
 * R P C _ D G _ S C A L L _ R E L E A S E _ N O _ U N L O C K
 *
 * Like RPC_DG_SCALL_RELEASE, except doesn't unlock the SCALL.  Note
 * that the referencing counting model requires that this macro can be
 * used iff the release will not be the "last one" (i.e., the one that
 * would normally cause the SCALL to be freed).
 */

#define RPC_DG_SCALL_RELEASE_NO_UNLOCK(scall) { \
    RPC_DG_CALL_LOCK_ASSERT(&(*(scall))->c); \
    assert((*(scall))->c.refcnt > 1); \
    --(*(scall))->c.refcnt; \
    *(scall) = NULL; \
}

PRIVATE void rpc__dg_scall_free (rpc_dg_scall_p_t  /*scall*/);

PRIVATE void rpc__dg_scall_reinit (
        rpc_dg_scall_p_t  /*scall*/,
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/
    );

PRIVATE rpc_dg_scall_p_t rpc__dg_scall_alloc (
        rpc_dg_sct_elt_p_t  /*scte*/,
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/
    );

PRIVATE rpc_dg_scall_p_t rpc__dg_scall_cbk_alloc (
        rpc_dg_ccall_p_t  /*ccall*/,
        rpc_dg_sock_pool_elt_p_t  /*sp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/
    );

PRIVATE void rpc__dg_scall_orphan_call (
	rpc_dg_scall_p_t  /*scall*/
    );

#endif /* _DGSCALL_H */
