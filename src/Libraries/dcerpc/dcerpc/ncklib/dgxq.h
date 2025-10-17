/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
**      dgxq.h
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

#ifndef _DGXQ_H
#define _DGXQ_H

/*
 * R P C _ D G _ X M I T Q _ R E I N I T
 *
 * Reinitialize a transmit queue.
 */

#define RPC_DG_XMITQ_REINIT(xq, call) { \
    if ((xq)->head != NULL) rpc__dg_xmitq_free(xq, call); \
    rpc__dg_xmitq_reinit(xq); \
}

/*
 * R P C _ D G _ X M I T Q _ A W A I T I N G _ A C K _ S E T
 *
 * Mark a transmit queue as wanting an acknowledgement event (fack, response,
 * working, etc.)
 */

#define RPC_DG_XMITQ_AWAITING_ACK_SET(xq) ( \
    (xq)->awaiting_ack = true, \
    xq->awaiting_ack_timestamp = rpc__clock_stamp() \
)

/*
 * R P C _ D G _ X M I T Q _ A W A I T I N G _ A C K _ C L R
 *
 * Mark a transmit queue as no longer wanting an acknowledgement event.
 */

#define RPC_DG_XMITQ_AWAITING_ACK_CLR(xq) ( \
    (xq)->awaiting_ack = false \
)

PRIVATE void rpc__dg_xmitq_elt_xmit (
        rpc_dg_xmitq_elt_p_t  /*xqe*/,
        rpc_dg_call_p_t  /*call*/,
        boolean32  /*block*/
    );

PRIVATE void rpc__dg_xmitq_init (
        rpc_dg_xmitq_p_t  /*xq*/
    );

PRIVATE void rpc__dg_xmitq_reinit (
        rpc_dg_xmitq_p_t  /*xq*/
    );

PRIVATE void rpc__dg_xmitq_free (
        rpc_dg_xmitq_p_t  /*xq*/,
        rpc_dg_call_p_t  /*call*/
    );

PRIVATE void rpc__dg_xmitq_append_pp (
        rpc_dg_call_p_t  /*call*/,
        unsigned32 * /*st*/
    );

PRIVATE boolean rpc__dg_xmitq_awaiting_ack_tmo (
        rpc_dg_xmitq_p_t  /*xq*/,
        unsigned32  /*com_timeout_knob*/
    );

PRIVATE void rpc__dg_xmitq_restart (
        rpc_dg_call_p_t  /*call*/
    );

#endif /* _DGXQ_H */
