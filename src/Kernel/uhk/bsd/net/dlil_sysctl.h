/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
 * NOTICE: This file was modified by SPARTA, Inc. in 2005 to introduce
 * support for mandatory and extensible security protections.  This notice
 * is included in support of clause 2.2 (b) of the Apple Public License,
 * Version 2.0.
 */
#ifndef DLIL_SYSCTL_H
#define DLIL_SYSCTL_H

/*
 * Sysctl settings and metrics for DLIL.
 */

#if BSD_KERNEL_PRIVATE

#include <sys/kernel_types.h>
#include <net/if.h>
#include <net/dlil_var_private.h>


/******************************************************************************
* Section: DLIL send and receive queues.                                     *
******************************************************************************/

#define IF_RCVQ_MINLEN  32
#define IF_RCVQ_MAXLEN  256

extern uint32_t if_sndq_maxlen;
extern uint32_t if_rcvq_maxlen;
extern uint32_t if_delaybased_queue; /* enable/disable*/
extern uint32_t ifnet_start_delayed;
extern uint32_t ifnet_delay_start_disabled;
extern uint32_t if_rcvq_burst_limit;
extern uint32_t if_rcvq_trim_pct;
extern struct chain_len_stats tx_chain_len_stats;
extern uint32_t tx_chain_len_count;


/******************************************************************************
* Section: DLIL opportunistic rx polling.                                    *
******************************************************************************/

/* Input poll interval definitions */
#define IF_RXPOLL_INTERVALTIME_MIN      (1ULL * 1000)           /* 1 us */
#define IF_RXPOLL_INTERVALTIME          (1ULL * 1000 * 1000)    /* 1 ms */

extern uint32_t if_rxpoll;                   /* enable/disable */
extern uint32_t if_rxpoll_decay;
extern uint64_t if_rxpoll_mode_holdtime;
extern uint64_t if_rxpoll_sample_holdtime;
extern uint64_t if_rxpoll_interval_time;
extern uint32_t if_rxpoll_interval_pkts;
extern uint32_t if_sysctl_rxpoll_wlowat;
extern uint32_t if_sysctl_rxpoll_whiwat;
extern uint32_t if_rxpoll_max;

#if TEST_INPUT_THREAD_TERMINATION
extern uint32_t if_input_thread_termination_spin;
#endif /* TEST_INPUT_THREAD_TERMINATION */

extern uint32_t cur_dlil_input_threads;


/******************************************************************************
* Section: hardware-assisted checksum mechanism.                             *
******************************************************************************/

extern uint32_t hwcksum_tx;                  /* enable/disable */
extern uint32_t hwcksum_rx;                  /* enable/disable */
extern uint64_t hwcksum_in_invalidated;      /* Inbound packets with invalid hw cksum. */


/*
 * Hardware-assisted checksum debugging metrics.
 */

#define HWCKSUM_DBG_PARTIAL_FORCED      0x1  /* Forced partial checksum. */
#define HWCKSUM_DBG_PARTIAL_RXOFF_ADJ   0x2  /* Adjust start offset. */
#define HWCKSUM_DBG_FINALIZE_FORCED     0x10 /* Forced finalize. */
#define HWCKSUM_DBG_MASK \
	(HWCKSUM_DBG_PARTIAL_FORCED | HWCKSUM_DBG_PARTIAL_RXOFF_ADJ |   \
	HWCKSUM_DBG_FINALIZE_FORCED)

extern uint32_t hwcksum_dbg;                        /* enable/disable */
extern uint32_t hwcksum_dbg_mode;                   /* HWCKSUM_DBG_ bitmask */
extern uint64_t hwcksum_dbg_partial_forced;         /* Packets forced using partial cksum. */
extern uint64_t hwcksum_dbg_partial_forced_bytes;   /* Bytes forced using partial cksum. */
extern uint32_t hwcksum_dbg_partial_rxoff_forced;   /* Forced partial cksum rx offset. */
extern uint32_t hwcksum_dbg_partial_rxoff_adj;      /* Adjusted partial cksum rx offset. */
extern uint64_t hwcksum_dbg_verified;               /* Packets verified for having good cksum. */
extern uint64_t hwcksum_dbg_bad_cksum;              /* Packets with bad hw cksum. */
extern uint64_t hwcksum_dbg_bad_rxoff;              /* Packets with invalid rx offset. */
extern uint64_t hwcksum_dbg_adjusted;               /* Packets with adjusted rx offset. */
extern uint64_t hwcksum_dbg_finalized_hdr;          /* Finalized headers. */
extern uint64_t hwcksum_dbg_finalized_data;         /* Finalized payloads. */


/******************************************************************************
* Section: DLIL debugging, notifications and sanity checks                   *
******************************************************************************/

extern uint32_t if_flowadv;                 /* enable/disable */
extern uint32_t threshold_notify;           /* enable/disable */
extern uint32_t threshold_interval;         /* in seconds */
extern struct net_api_stats net_api_stats;
extern int dlil_verbose;
extern uint32_t net_wake_pkt_debug;

#if IFNET_INPUT_SANITY_CHK
extern uint32_t dlil_input_sanity_check;
#endif /* IFNET_INPUT_SANITY_CHK */



#endif /* BSD_KERNEL_PRIVATE */

#endif /* DLIL_SYSCTL_H */
