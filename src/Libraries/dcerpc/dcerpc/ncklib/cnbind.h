/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
**      cnbind.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Interface to the NCA Connection Protocol Service's Binding Service.
**
**
*/

#ifndef _CNBIND_H
#define _CNBIND_H	1

/*
 * R P C _ _ C N _ B I N D I N G _ A L L O C
 */

PRIVATE rpc_binding_rep_t *rpc__cn_binding_alloc (
    boolean32            /* is_server */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ I N I T
 */

PRIVATE void rpc__cn_binding_init (
    rpc_binding_rep_p_t  /* binding_r */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ R E S E T
 */

PRIVATE void rpc__cn_binding_reset (
    rpc_binding_rep_p_t  /* binding_r */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ C H A N G E D
 */

PRIVATE void rpc__cn_binding_changed (
    rpc_binding_rep_p_t  /* binding_r */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ F R E E
 */

PRIVATE void rpc__cn_binding_free (
    rpc_binding_rep_p_t * /* binding_r */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ I N Q _ A D D R
 */

PRIVATE void rpc__cn_binding_inq_addr (
    rpc_binding_rep_p_t  /* binding_r */,
    rpc_addr_p_t        * /* rpc_addr */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ I N Q _ C L I E N T
 */

PRIVATE void rpc__cn_binding_inq_client (
    rpc_binding_rep_p_t  /* binding_r */,
    rpc_client_handle_t * /* client_h */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ C O P Y
 */

PRIVATE void rpc__cn_binding_copy (
    rpc_binding_rep_p_t  /* src_binding_r */,
    rpc_binding_rep_p_t  /* dst_binding_r */,
    unsigned32          * /* st */);

/*
 * R P C _ _ C N _ B I N D I N G _ C R O S S _ F O R K
 */

PRIVATE void rpc__cn_binding_cross_fork (
    rpc_binding_rep_p_t  /* binding_r */,
    unsigned32          * /* st */);

#endif /* _CNBIND_H */
