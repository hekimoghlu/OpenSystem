/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
**      dgutl.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Utility routines for the NCA RPC datagram protocol implementation.
**
**
*/

#ifndef _DGUTL_H
#define _DGUTL_H

/* ========================================================================= */

#ifndef RPC_DG_PLOG

#define RPC_DG_PLOG_RECVFROM_PKT(hdrp, bodyp)
#define RPC_DG_PLOG_SENDMSG_PKT(iov, iovlen)
#define RPC_DG_PLOG_LOSSY_SENDMSG_PKT(iov, iovlen, lossy_action)
#define rpc__dg_plog_pkt(hdrp, bodyp, recv, lossy_action)

#else

#define RPC_DG_PLOG_RECVFROM_PKT(hdrp, bodyp) \
    { \
        if (RPC_DBG(rpc_es_dbg_dg_pktlog, 100)) \
            rpc__dg_plog_pkt((hdrp), (bodyp), true, 0); \
    }

#define RPC_DG_PLOG_SENDMSG_PKT(iov, iovlen) \
    { \
        if (RPC_DBG(rpc_es_dbg_dg_pktlog, 100)) \
            rpc__dg_plog_pkt((rpc_dg_raw_pkt_hdr_p_t) (iov)[0].base,  \
                    (iovlen) < 2 ? NULL : (rpc_dg_pkt_body_p_t) (iov)[1].base,  \
                    false, 3); \
    }

#define RPC_DG_PLOG_LOSSY_SENDMSG_PKT(iov, iovlen, lossy_action) \
    { \
        if (RPC_DBG(rpc_es_dbg_dg_pktlog, 100)) \
            rpc__dg_plog_pkt((rpc_dg_raw_pkt_hdr_p_t) (iov)[0].base,  \
                    (iovlen) < 2 ? NULL : (rpc_dg_pkt_body_p_t) (iov)[1].base,  \
                    false, lossy_action); \
    }

PRIVATE void rpc__dg_plog_pkt (
        rpc_dg_raw_pkt_hdr_p_t  /*hdrp*/,
        rpc_dg_pkt_body_p_t  /*bodyp*/,
        boolean32  /*recv*/,
        unsigned32  /*lossy_action*/
    );

PRIVATE void rpc__dg_plog_dump (
         /*void*/
    );

#endif

/* ========================================================================= */

PRIVATE void rpc__dg_xmit_pkt (
        rpc_socket_t  /*sock*/,
        rpc_addr_p_t  /*addr*/,
        rpc_socket_iovec_p_t  /*iov*/,
        int  /*iovlen*/,
        boolean * /*b*/
    );

PRIVATE void rpc__dg_xmit_hdr_only_pkt (
        rpc_socket_t  /*sock*/,
        rpc_addr_p_t  /*addr*/,
        rpc_dg_pkt_hdr_p_t  /*hdrp*/,
        rpc_dg_ptype_t  /*ptype*/
    );

PRIVATE void rpc__dg_xmit_error_body_pkt (
        rpc_socket_t  /*sock*/,
        rpc_addr_p_t  /*addr*/,
        rpc_dg_pkt_hdr_p_t  /*hdrp*/,
        rpc_dg_ptype_t  /*ptype*/,
        unsigned32  /*errst*/
    );

PRIVATE const char *rpc__dg_act_seq_string (
        rpc_dg_pkt_hdr_p_t  /*hdrp*/
    );

PRIVATE const char *rpc__dg_pkt_name (
        rpc_dg_ptype_t  /*ptype*/
    );

PRIVATE unsigned16 rpc__dg_uuid_hash (
        uuid_p_t  /*uuid*/
    );

#endif /* _DGUTL_H */
