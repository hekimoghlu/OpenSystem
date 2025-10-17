/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
**      dgrq.h
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

#ifndef _DGRQ_H
#define _DGRQ_H

/*
 * R P C _ D G _ R E C V Q _ E L T _ F R O M _ I O V E C T O R _ E L T
 *
 * Given an IO vector element, return the associated receive queue element.
 * This macro is used by internal callers of the "comm_receive/transceive"
 * path so that can get back a receieve queue element and look at the
 * header.  This macro depends on how "comm_receive" works.
 */

#define RPC_DG_RECVQ_ELT_FROM_IOVECTOR_ELT(iove) \
    ((iove)->buff_addr)

/*
 * R P C _ D G _ R E C V Q _ R E I N I T
 *
 * Reinitialize a receive queue.
 */

#define RPC_DG_RECVQ_REINIT(rq) { \
    if ((rq)->head != NULL) rpc__dg_recvq_free(rq); \
    rpc__dg_recvq_init(rq);     /* !!! Maybe be smarter later -- this may be losing "history" */ \
}

/*
 * R P C _ D G _ R E C V Q _ I O V E C T O R _ S E T U P
 *
 * Setup the return iovector element.
 *
 * NOTE WELL that other logic depends on the fact that the "buff_addr"
 * field of iovector elements points to an "rpc_dg_recvq_elt_t" (rqe).
 * See comments by RPC_DG_RECVQ_ELT_FROM_IOVECTOR_ELT.
 */

#define RPC_DG_RECVQ_IOVECTOR_SETUP(data, rqe) { \
    (data)->buff_dealloc  = (rpc_buff_dealloc_fn_t) rpc__dg_pkt_free_rqe_for_stub; \
    (data)->buff_addr     = (byte_p_t) (rqe); \
    (data)->buff_len      = sizeof *(rqe); \
    (data)->data_addr     = (byte_p_t) &(rqe)->pkt->body; \
    (data)->data_len      = ((rqe)->hdrp != NULL) ? \
                                MIN((rqe)->hdrp->len, \
                                    (rqe)->pkt_len - RPC_C_DG_RAW_PKT_HDR_SIZE) : \
                                (rqe)->pkt_len; \
}

PRIVATE void rpc__dg_recvq_init ( rpc_dg_recvq_p_t  /*rq*/);

PRIVATE void rpc__dg_recvq_free ( rpc_dg_recvq_p_t  /*rq*/);

#endif /* _DGRQ_H */
