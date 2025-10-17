/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#ifndef _UAPI_LINUX_TCP_METRICS_H
#define _UAPI_LINUX_TCP_METRICS_H
#include <linux/types.h>
#define TCP_METRICS_GENL_NAME "tcp_metrics"
#define TCP_METRICS_GENL_VERSION 0x1
enum tcp_metric_index {
  TCP_METRIC_RTT,
  TCP_METRIC_RTTVAR,
  TCP_METRIC_SSTHRESH,
  TCP_METRIC_CWND,
  TCP_METRIC_REORDERING,
  TCP_METRIC_RTT_US,
  TCP_METRIC_RTTVAR_US,
  __TCP_METRIC_MAX,
};
#define TCP_METRIC_MAX (__TCP_METRIC_MAX - 1)
enum {
  TCP_METRICS_A_METRICS_RTT = 1,
  TCP_METRICS_A_METRICS_RTTVAR,
  TCP_METRICS_A_METRICS_SSTHRESH,
  TCP_METRICS_A_METRICS_CWND,
  TCP_METRICS_A_METRICS_REODERING,
  TCP_METRICS_A_METRICS_RTT_US,
  TCP_METRICS_A_METRICS_RTTVAR_US,
  __TCP_METRICS_A_METRICS_MAX
};
#define TCP_METRICS_A_METRICS_MAX (__TCP_METRICS_A_METRICS_MAX - 1)
enum {
  TCP_METRICS_ATTR_UNSPEC,
  TCP_METRICS_ATTR_ADDR_IPV4,
  TCP_METRICS_ATTR_ADDR_IPV6,
  TCP_METRICS_ATTR_AGE,
  TCP_METRICS_ATTR_TW_TSVAL,
  TCP_METRICS_ATTR_TW_TS_STAMP,
  TCP_METRICS_ATTR_VALS,
  TCP_METRICS_ATTR_FOPEN_MSS,
  TCP_METRICS_ATTR_FOPEN_SYN_DROPS,
  TCP_METRICS_ATTR_FOPEN_SYN_DROP_TS,
  TCP_METRICS_ATTR_FOPEN_COOKIE,
  TCP_METRICS_ATTR_SADDR_IPV4,
  TCP_METRICS_ATTR_SADDR_IPV6,
  TCP_METRICS_ATTR_PAD,
  __TCP_METRICS_ATTR_MAX,
};
#define TCP_METRICS_ATTR_MAX (__TCP_METRICS_ATTR_MAX - 1)
enum {
  TCP_METRICS_CMD_UNSPEC,
  TCP_METRICS_CMD_GET,
  TCP_METRICS_CMD_DEL,
  __TCP_METRICS_CMD_MAX,
};
#define TCP_METRICS_CMD_MAX (__TCP_METRICS_CMD_MAX - 1)
#endif
