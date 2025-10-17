/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#ifndef __LINUX_GEN_STATS_H
#define __LINUX_GEN_STATS_H
#include <linux/types.h>
enum {
  TCA_STATS_UNSPEC,
  TCA_STATS_BASIC,
  TCA_STATS_RATE_EST,
  TCA_STATS_QUEUE,
  TCA_STATS_APP,
  TCA_STATS_RATE_EST64,
  TCA_STATS_PAD,
  TCA_STATS_BASIC_HW,
  TCA_STATS_PKT64,
  __TCA_STATS_MAX,
};
#define TCA_STATS_MAX (__TCA_STATS_MAX - 1)
struct gnet_stats_basic {
  __u64 bytes;
  __u32 packets;
};
struct gnet_stats_rate_est {
  __u32 bps;
  __u32 pps;
};
struct gnet_stats_rate_est64 {
  __u64 bps;
  __u64 pps;
};
struct gnet_stats_queue {
  __u32 qlen;
  __u32 backlog;
  __u32 drops;
  __u32 requeues;
  __u32 overlimits;
};
struct gnet_estimator {
  signed char interval;
  unsigned char ewma_log;
};
#endif
