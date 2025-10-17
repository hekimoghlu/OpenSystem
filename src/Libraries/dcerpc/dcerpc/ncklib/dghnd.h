/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
**      dghnd.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Include file for "dghnd.c".
**
**
*/

#ifndef _DGHND_H
#define _DGHND_H

/*
 * R P C _ D G _ B I N D I N G _ S E R V E R _ R E I N I T
 *
 * Reinitialize the per-call variant information in a cached handle.
 * The binding handle is bound to the scall; the scall is bound to an
 * actid; the actid is bound to the auth identity - so the auth identity
 * can't change.  The object id can change.  The address of the client
 * can change (actid's aren't even bound to a protseq let alone specific
 * network-addr/endpoint pairs; sigh).
 */
#define RPC_DG_BINDING_SERVER_REINIT(h) { \
    unsigned32 st; \
    \
    (h)->c.c.obj = (h)->scall->c.call_object; \
    rpc__naf_addr_overcopy((h)->scall->c.addr, &(h)->c.c.rpc_addr, &st); \
}

/* ========================================================================= */

rpc_binding_rep_t *rpc__dg_binding_alloc    (
        boolean32  /*is_server*/,
        unsigned32 * /*st*/
    );
void rpc__dg_binding_init    (
        rpc_binding_rep_p_t  /*h*/,
        unsigned32 * /*st*/
    );
void rpc__dg_binding_reset    (
        rpc_binding_rep_p_t  /*h*/,
        unsigned32 * /*st*/
    );
void rpc__dg_binding_changed    (
        rpc_binding_rep_p_t  /*h*/,
        unsigned32 * /*st*/
    );
void rpc__dg_binding_free    (
        rpc_binding_rep_p_t * /*h*/,
        unsigned32 * /*st*/
    );
void rpc__dg_binding_inq_addr    (
        rpc_binding_rep_p_t  /*h*/,
        rpc_addr_p_t * /*rpc_addr*/,
        unsigned32 * /*st*/
    );
void rpc__dg_binding_copy    (
        rpc_binding_rep_p_t  /*src_h*/,
        rpc_binding_rep_p_t  /*dst_h*/,
        unsigned32 * /*st*/
    );

rpc_dg_binding_client_p_t rpc__dg_binding_srvr_to_client    (
        rpc_dg_binding_server_p_t  /*shand*/,
        unsigned32 * /*st*/
    );

void rpc__dg_binding_cross_fork    (
        rpc_binding_rep_p_t  /*h*/,
        unsigned32 * /*st*/
    );

#endif /* _DGHND_H */
