/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
**      noauthdgp.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Definition of types private to the noauth-datagram glue module.
**
**
*/

#ifndef _NOAUTHDG_H
#define _NOAUTHDG_H	1

#define NCK_NEED_MARSHALLING

#include <dg.h>
#include <noauth.h>
#include <dce/conv.h>

/*
 * For various reasons, it's painful to get at the NDR tag of the
 * underlying data, so we cheat and just encode it in big-endian order.
 */

#define rpc_marshall_be_long_int(mp, bei) \
{       long temp = htonl(bei);            \
        rpc_marshall_long_int (mp, temp);  \
}

#define rpc_convert_be_long_int(mp, bei) \
{                                       \
    rpc_unmarshall_long_int(mp, bei);   \
    bei = ntohl(bei);                   \
}

#define rpc_marshall_be_short_int(mp, bei) \
{       short temp = htons(bei);            \
        rpc_marshall_short_int (mp, temp);  \
}

#define rpc_convert_be_short_int(mp, bei) \
{                                       \
    rpc_unmarshall_short_int(mp, bei);   \
    bei = ntohs(bei);                   \
}


/*
 * DG EPV routines.
 */

#ifdef __cplusplus
extern "C" {
#endif

void rpc__noauth_dg_pre_call (
        rpc_auth_info_p_t               ,
        handle_t                        ,
        unsigned32                      *
    );

rpc_auth_info_p_t rpc__noauth_dg_create (
        unsigned32                      * /*st*/
    );

void rpc__noauth_dg_encrypt (
        rpc_auth_info_p_t                /*info*/,
        rpc_dg_xmitq_elt_p_t            ,
        unsigned32                      * /*st*/
    );

void rpc__noauth_dg_pre_send (
        rpc_auth_info_p_t                /*info*/,
        rpc_dg_xmitq_elt_p_t             /*pkt*/,
        rpc_dg_pkt_hdr_p_t               /*hdrp*/,
        rpc_socket_iovec_p_t             /*iov*/,
        int                              /*iovlen*/,
        dce_pointer_t                        /*cksum*/,
        unsigned32                      * /*st*/
    );

void rpc__noauth_dg_recv_ck (
        rpc_auth_info_p_t                /*info*/,
        rpc_dg_recvq_elt_p_t             /*pkt*/,
        dce_pointer_t                        /*cksum*/,
        error_status_t                  * /*st*/
    );

void rpc__noauth_dg_who_are_you (
        rpc_auth_info_p_t                /*info*/,
        handle_t                        ,
        idl_uuid_t                          *,
        unsigned32                      ,
        unsigned32                      *,
        idl_uuid_t                          *,
        unsigned32                      *
    );

void rpc__noauth_dg_way_handler (
        rpc_auth_info_p_t                /*info*/,
        ndr_byte                        * /*in_data*/,
        signed32                         /*in_len*/,
        signed32                         /*out_max_len*/,
        ndr_byte                        * /*out_data*/,
        signed32                        * /*out_len*/,
        unsigned32                      * /*st*/
    );

#ifdef __cplusplus
}
#endif

#endif /* _NOAUTHDG_H */
