/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
**      dgcct.h
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

#ifndef _DGCCT_H
#define _DGCCT_H

/*
 * R P C _ D G _ C C T E _ R E F _ I N I T
 *
 * Initialize a CCTE soft reference.
 */

#define RPC_DG_CCTE_REF_INIT(rp) ( \
    (rp)->ccte = NULL, \
    (rp)->gc_count = 0 \
)

/*
 * R P C _ D G _ C C T _ R E F E R E N C E
 *
 * Increment the reference count for the CCTE.
 */

#define RPC_DG_CCT_REFERENCE(ccte) { \
    assert((ccte)->refcnt < 255); \
    (ccte)->refcnt++; \
}

/*
 * R P C _ D G _ C C T _ R E L E A S E
 *
 * Release a CCTE and update its last time used (for CCT GC aging).
 * Retain the soft reference so we can reuse it on subsequent calls.
 */

#define RPC_DG_CCT_RELEASE(ccte_ref) { \
    rpc_dg_cct_elt_p_t ccte = (ccte_ref)->ccte; \
    assert(ccte->refcnt > 1); \
    if (--ccte->refcnt <= 1) \
        ccte->timestamp = rpc__clock_stamp(); \
}

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__dg_cct_get (
        rpc_auth_info_p_t /*auth_info*/,
        rpc_dg_ccall_p_t /*ccall*/
    );

PRIVATE void rpc__dg_cct_release (
        rpc_dg_ccte_ref_p_t /*ccte_ref*/
    );

PRIVATE void rpc__dg_cct_fork_handler (
        rpc_fork_stage_id_t /*stage*/
    );

#ifdef __cplusplus
}
#endif

#endif /* _DGCCT_H */
