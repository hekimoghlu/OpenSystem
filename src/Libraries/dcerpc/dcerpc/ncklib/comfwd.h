/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
**      comfwd.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Private interface to the Common Communications Forwarding Service for use
**  by RPC Protocol Services and Local Location Broker.  This service is
**  in its own file (rather then com.h) so that the llb does not have to
**  include other runtime internal include files.
**
**
*/

#ifndef _COMFWD_H
#define _COMFWD_H	1

/***********************************************************************/
/*
 * The beginning of this file specifies the "public" (visible to the llbd)
 * portion of the fwd interface.
 */

/*
 * The signature of a server forwarding map function.
 *
 * The function determines an appropriate forwarding address for the
 * packet based on the provided info.  The "fwd_action" output parameter
 * determines the disposition of the packet; "drop" means just drop the
 * packet (don't forward, don't send anything back to the client), "reject"
 * means send a rejection back to the client (and drop the packet), and
 * "forward" means that the packet should be forwarded to the address
 * specified in the "fwd_addr" output parameter.
 *
 * WARNING:  This should be a relatively light weight, non-blocking
 * function as the runtime may (will likely) be calling the function
 * from the context of the runtime's listener thread (i.e. handling
 * of additional incoming received packets is suspended while this
 * provided function is executing).
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    rpc_e_fwd_drop,
    rpc_e_fwd_reject,
    rpc_e_fwd_forward,
    rpc_e_fwd_delayed
} rpc_fwd_action_t;

typedef void (*rpc_fwd_map_fn_t) (
        /* [in] */    uuid_p_t           /*obj_uuid*/,
        /* [in] */    rpc_if_id_p_t      /*if_id*/,
        /* [in] */    rpc_syntax_id_p_t  /*data_rep*/,
        /* [in] */    rpc_protocol_id_t  /*rpc_protocol*/,
        /* [in] */    unsigned32         /*rpc_protocol_vers_major*/,
        /* [in] */    unsigned32         /*rpc_protocol_vers_minor*/,
        /* [in] */    rpc_addr_p_t       /*addr*/,
        /* [in] */    uuid_p_t           /*actuuid*/,
        /* [out] */   rpc_addr_p_t      * /*fwd_addr*/,
        /* [out] */   rpc_fwd_action_t  * /*fwd_action*/,
        /* [out] */   unsigned32        * /*status*/
    );

/*
 * Register a forwarding map function with the runtime.  This registered
 * function will be called by the protocol services to determine an
 * appropriate forwarding endpoint for a received pkt that is not for
 * any of the server's registered interfaces.
 */
PRIVATE void rpc__server_register_fwd_map (
        /* [in] */    rpc_fwd_map_fn_t    /*map_fn*/,
        /* [out] */   unsigned32          * /*status*/
    );

#if 0
/* Removed unused symbol for rdar://problem/26430747 */
PRIVATE void rpc__server_fwd_resolve_delayed (
	/* [in] */   uuid_p_t            /*actuuid*/,
        /* [in] */   rpc_addr_p_t        /*fwd_addr*/,
        /* [in] */   rpc_fwd_action_t  * /*fwd_action*/,
        /* [out] */  unsigned32        * /*status*/
    );
#endif
/***********************************************************************/
/*
 * The following are to be considered internal to the runtime.
 */

/*
 * R P C _ G _ F W D _ F N
 *
 * The global forwarding map function variable.  Its value indicates
 * whether or not the RPC runtime should be performing forwarding services
 * and if so, the forwarding map function to use. Its definition is in comp.c.
 */
EXTERNAL rpc_fwd_map_fn_t       rpc_g_fwd_fn;

#endif /* _COMFWD_H_ */
