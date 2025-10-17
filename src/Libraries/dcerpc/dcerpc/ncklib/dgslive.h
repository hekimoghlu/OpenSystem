/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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
**      dgslive.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG maintain/monitor liveness service routines.
**
**
*/

#ifndef _DGSLIVE_H
#define _DGSLIVE_H

/*
 * R P C _ D G _ C L I E N T _ R E L E A S E
 *
 * Release a reference to a client handle.  This macro is called by the SCT
 * monitor when it is time to free an SCT entry.  It the entry has a reference
 * to a client handle, that reference is decremented.  If the count falls to 1,
 * meaning the only other reference is the client_rep table, the client_free
 * routine is called to free the handle.
 */

#define RPC_DG_CLIENT_RELEASE(scte) { \
    if ((scte)->client != NULL) \
    { \
        rpc_dg_client_rep_p_t client = (scte)->client; \
        assert(client->refcnt > 1); \
        if (--client->refcnt == 1) \
            rpc__dg_client_free((rpc_client_handle_t) (scte)->client);\
        (scte)->client = NULL; \
    } \
}

PRIVATE void rpc__dg_binding_inq_client (
        rpc_binding_rep_p_t  /*binding_r*/,
        rpc_client_handle_t * /*client_h*/,
        unsigned32 * /*st*/
    );

PRIVATE void rpc__dg_client_free (
        rpc_client_handle_t  /*client_h*/
    );

#endif /* _DGSLIVE_H */
