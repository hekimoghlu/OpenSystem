/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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
#ifndef _NET_TIMESTAMPING_H
#define _NET_TIMESTAMPING_H
#include <linux/types.h>
#include <linux/socket.h>
enum {
  SOF_TIMESTAMPING_TX_HARDWARE = (1 << 0),
  SOF_TIMESTAMPING_TX_SOFTWARE = (1 << 1),
  SOF_TIMESTAMPING_RX_HARDWARE = (1 << 2),
  SOF_TIMESTAMPING_RX_SOFTWARE = (1 << 3),
  SOF_TIMESTAMPING_SOFTWARE = (1 << 4),
  SOF_TIMESTAMPING_SYS_HARDWARE = (1 << 5),
  SOF_TIMESTAMPING_RAW_HARDWARE = (1 << 6),
  SOF_TIMESTAMPING_OPT_ID = (1 << 7),
  SOF_TIMESTAMPING_TX_SCHED = (1 << 8),
  SOF_TIMESTAMPING_TX_ACK = (1 << 9),
  SOF_TIMESTAMPING_OPT_CMSG = (1 << 10),
  SOF_TIMESTAMPING_OPT_TSONLY = (1 << 11),
  SOF_TIMESTAMPING_OPT_STATS = (1 << 12),
  SOF_TIMESTAMPING_OPT_PKTINFO = (1 << 13),
  SOF_TIMESTAMPING_OPT_TX_SWHW = (1 << 14),
  SOF_TIMESTAMPING_BIND_PHC = (1 << 15),
  SOF_TIMESTAMPING_OPT_ID_TCP = (1 << 16),
  SOF_TIMESTAMPING_OPT_RX_FILTER = (1 << 17),
  SOF_TIMESTAMPING_LAST = SOF_TIMESTAMPING_OPT_RX_FILTER,
  SOF_TIMESTAMPING_MASK = (SOF_TIMESTAMPING_LAST - 1) | SOF_TIMESTAMPING_LAST
};
#define SOF_TIMESTAMPING_TX_RECORD_MASK (SOF_TIMESTAMPING_TX_HARDWARE | SOF_TIMESTAMPING_TX_SOFTWARE | SOF_TIMESTAMPING_TX_SCHED | SOF_TIMESTAMPING_TX_ACK)
struct so_timestamping {
  int flags;
  int bind_phc;
};
struct hwtstamp_config {
  int flags;
  int tx_type;
  int rx_filter;
};
enum hwtstamp_flags {
  HWTSTAMP_FLAG_BONDED_PHC_INDEX = (1 << 0),
#define HWTSTAMP_FLAG_BONDED_PHC_INDEX HWTSTAMP_FLAG_BONDED_PHC_INDEX
  HWTSTAMP_FLAG_LAST = HWTSTAMP_FLAG_BONDED_PHC_INDEX,
  HWTSTAMP_FLAG_MASK = (HWTSTAMP_FLAG_LAST - 1) | HWTSTAMP_FLAG_LAST
};
enum hwtstamp_tx_types {
  HWTSTAMP_TX_OFF,
  HWTSTAMP_TX_ON,
  HWTSTAMP_TX_ONESTEP_SYNC,
  HWTSTAMP_TX_ONESTEP_P2P,
  __HWTSTAMP_TX_CNT
};
enum hwtstamp_rx_filters {
  HWTSTAMP_FILTER_NONE,
  HWTSTAMP_FILTER_ALL,
  HWTSTAMP_FILTER_SOME,
  HWTSTAMP_FILTER_PTP_V1_L4_EVENT,
  HWTSTAMP_FILTER_PTP_V1_L4_SYNC,
  HWTSTAMP_FILTER_PTP_V1_L4_DELAY_REQ,
  HWTSTAMP_FILTER_PTP_V2_L4_EVENT,
  HWTSTAMP_FILTER_PTP_V2_L4_SYNC,
  HWTSTAMP_FILTER_PTP_V2_L4_DELAY_REQ,
  HWTSTAMP_FILTER_PTP_V2_L2_EVENT,
  HWTSTAMP_FILTER_PTP_V2_L2_SYNC,
  HWTSTAMP_FILTER_PTP_V2_L2_DELAY_REQ,
  HWTSTAMP_FILTER_PTP_V2_EVENT,
  HWTSTAMP_FILTER_PTP_V2_SYNC,
  HWTSTAMP_FILTER_PTP_V2_DELAY_REQ,
  HWTSTAMP_FILTER_NTP_ALL,
  __HWTSTAMP_FILTER_CNT
};
struct scm_ts_pktinfo {
  __u32 if_index;
  __u32 pkt_length;
  __u32 reserved[2];
};
enum txtime_flags {
  SOF_TXTIME_DEADLINE_MODE = (1 << 0),
  SOF_TXTIME_REPORT_ERRORS = (1 << 1),
  SOF_TXTIME_FLAGS_LAST = SOF_TXTIME_REPORT_ERRORS,
  SOF_TXTIME_FLAGS_MASK = (SOF_TXTIME_FLAGS_LAST - 1) | SOF_TXTIME_FLAGS_LAST
};
struct sock_txtime {
  __kernel_clockid_t clockid;
  __u32 flags;
};
#endif
