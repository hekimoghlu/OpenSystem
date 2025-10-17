/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
**      dgsct.h
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

#ifndef _DGSCT_H
#define _DGSCT_H

/*
 * R P C _ D G _ S C T _ I S _ W A Y _ V A L I D A T E D
 *
 * Return true only if the connection has a WAY validated seq and the
 * client doesn't require us to WAY it just to get it the server's boot
 * time.
 *
 * It's ok to look at the flag without the lock;  once it's set, it
 * never becomes unset - if the test fails we'll end up doing extra
 * work when we may not have needed to.
 */

#define RPC_DG_SCT_IS_WAY_VALIDATED(scte) \
( \
    (scte)->high_seq_is_way_validated && \
    ! (scte)->scall->client_needs_sboot \
)

/*
 * R P C _ D G _ S C T _ R E F E R E N C E
 *
 * Increment the reference count for the SCTE.
 */

#define RPC_DG_SCT_REFERENCE(scte) { \
    assert((scte)->refcnt < 255); \
    (scte)->refcnt++; \
}

/*
 * R P C _ D G _ S C T _ R E L E A S E
 *
 * Release a currently inuse SCTE.
 *
 * If the reference count goes to one, the SCTE is now available for
 * reuse / or GCing.  Update the SCTE's last used timestamp.
 */

#define RPC_DG_SCT_RELEASE(scte) { \
    RPC_LOCK_ASSERT(0); \
    assert((*(scte))->refcnt > 0); \
    if (--(*scte)->refcnt <= 1) \
        (*(scte))->timestamp = rpc__clock_stamp(); \
    *(scte) = NULL; \
}

PRIVATE void rpc__dg_sct_inq_scall (
        rpc_dg_sct_elt_p_t  /*scte*/,
        rpc_dg_scall_p_t * /*scallp*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/
    );

PRIVATE void rpc__dg_sct_new_call (
        rpc_dg_sct_elt_p_t  /*scte*/,
        rpc_dg_sock_pool_elt_p_t  /*si*/,
        rpc_dg_recvq_elt_p_t  /*rqe*/,
        rpc_dg_scall_p_t * /*scallp*/
    );

PRIVATE void rpc__dg_sct_backout_new_call (
        rpc_dg_sct_elt_p_t  /*scte*/,
        unsigned32  /*seq*/
    );

PRIVATE rpc_dg_sct_elt_p_t rpc__dg_sct_lookup (
        uuid_p_t  /*actid*/,
        unsigned32  /*probe_hint*/
    );

PRIVATE rpc_dg_sct_elt_p_t rpc__dg_sct_get (
        uuid_p_t  /*actid*/,
        unsigned32  /*probe_hint*/,
        unsigned32  /*seq*/
    );

PRIVATE rpc_binding_handle_t rpc__dg_sct_make_way_binding (
        rpc_dg_sct_elt_p_t  /*scte*/,
        unsigned32 * /*st*/
    );

PRIVATE void rpc__dg_sct_way_validate (
        rpc_dg_sct_elt_p_t  /*scte*/,
        unsigned32       /*force_way_auth*/,
        unsigned32      * /*st*/
    );

PRIVATE void rpc__dg_sct_fork_handler (
        rpc_fork_stage_id_t  /*stage*/
    );

#endif /* _DGSCT_H */
