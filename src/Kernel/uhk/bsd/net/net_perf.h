/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#ifndef _NET_NET_PERF_H_
#define _NET_NET_PERF_H_

#include <stdint.h>

#ifdef KERNEL_PRIVATE
#include <sys/time.h>
#include <mach/boolean.h>
#endif /* KERNEL_PRIVATE */

/* five histogram bins are separated by four dividing "bars" */
#define NET_PERF_BARS 4

typedef struct net_perf {
	uint64_t np_total_pkts; /* total packets input or output during measurement */
	uint64_t np_total_usecs;        /* microseconds elapsed during measurement */
	uint64_t np_hist1;              /* histogram bin 1 */
	uint64_t np_hist2;              /* histogram bin 2 */
	uint64_t np_hist3;              /* histogram bin 3 */
	uint64_t np_hist4;              /* histogram bin 4 */
	uint64_t np_hist5;              /* histogram bin 5 */
	uint8_t np_hist_bars[NET_PERF_BARS];
} net_perf_t;

#ifdef KERNEL_PRIVATE
void net_perf_initialize(net_perf_t *npp, uint64_t bins);
void net_perf_start_time(net_perf_t *npp, struct timeval *tv);
void net_perf_measure_time(net_perf_t *npp, struct timeval *start, uint64_t num_pkts);
void net_perf_histogram(net_perf_t *npp, uint64_t num_pkts);
boolean_t net_perf_validate_bins(uint64_t bins);

#endif /* KERNEL_PRIVATE */

#endif /* _NET_NET_PERF_H_ */
