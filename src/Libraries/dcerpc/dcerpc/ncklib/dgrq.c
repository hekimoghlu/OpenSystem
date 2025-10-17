/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
**      dgrq.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG protocol service routines.  Handles receive queues.
**
**
*/

#include <dg.h>
#include <dgrq.h>
#include <dgpkt.h>

/* ========================================================================= */

#define DEFAULT_WAKE_THREAD_QSIZE   (6 * 1024)

/* ========================================================================= */

/*
 * R P C _ _ D G _ R E C V Q _ I N I T
 *
 * Initialize a receive queue (rpc_dg_recv_q_t).
 */

PRIVATE void rpc__dg_recvq_init
(
    rpc_dg_recvq_p_t rq
)
{
    /*
     * presumably the call is either locked or 'private' at this point
     * RPC_DG_CALL_LOCK_ASSERT(call);
     */

    rq->head = rq->last_inorder = NULL;

    /*
     * high_rcv_frag_size should be set to zero. However, since its
     * primary use is a detection of the sender's MBF capability, we
     * start it from rpc_c_dg_initial_max_pkt_size for a little
     * performance improvement.
     */
    rq->high_rcv_frag_size = RPC_C_DG_INITIAL_MAX_PKT_SIZE;
    rq->next_fragnum    = 0;
    rq->high_fragnum    = -1;
    rq->high_serial_num = -1;
    rq->head_fragnum    = -1;
    rq->head_serial_num = -1;

    rq->wake_thread_qsize = DEFAULT_WAKE_THREAD_QSIZE;
    rq->max_queue_len = RPC_C_DG_MAX_RECVQ_LEN;

#ifdef DEBUG
    /*
     * For testing, allow an override via debug switch 10.
     */
    if (RPC_DBG (rpc_es_dbg_dg_rq_qsize, 1))
        rq->wake_thread_qsize = ((unsigned32)
            (rpc_g_dbg_switches[(int) rpc_es_dbg_dg_rq_qsize])) * 1024;
#endif

    rq->queue_len       = 0;
    rq->inorder_len     = 0;
    rq->recving_frags   = false;
    rq->all_pkts_recvd  = false;
    rq->is_way_validated= false;
}

/*
 * R P C _ _ D G _ R E C V Q _ F R E E
 *
 * Frees data referenced by a receive queue (rpc_dg_recv_q_t).  The
 * receive queue itself is NOT freed, since it's (assumed to be) part
 * of a larger structure.
 */

PRIVATE void rpc__dg_recvq_free
(
    rpc_dg_recvq_p_t rq
)
{
    /*
     * Presumably the call is either locked or 'private' at this point.
     * The NULL call handle passed to free_rqe() below, implies that we
     * are sure that this call is not currently blocked waiting for a
     * packet.
     *
     * RPC_DG_CALL_LOCK_ASSERT(call);
     */

    while (rq->head != NULL) {
        rpc_dg_recvq_elt_p_t rqe = rq->head;

        rq->head = rqe->next;
        rpc__dg_pkt_free_rqe(rqe, NULL);
    }
}
