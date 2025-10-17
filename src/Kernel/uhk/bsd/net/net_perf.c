/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
#include <net/if_var.h>
#include <net/net_perf.h>
#include <netinet/in_var.h>
#include <sys/sysctl.h>

static void ip_perf_record_stats(net_perf_t *npp, struct timeval *tv1,
    struct timeval *tv2, uint64_t num_pkts);
static void update_bins(net_perf_t *npp, uint64_t bins);

void
net_perf_start_time(net_perf_t *npp, struct timeval *tv)
{
#pragma unused(npp)
	microtime(tv);
}

void
net_perf_measure_time(net_perf_t *npp, struct timeval *start, uint64_t num_pkts)
{
	struct timeval stop;
	microtime(&stop);
	ip_perf_record_stats(npp, start, &stop, num_pkts);
}

static void
ip_perf_record_stats(net_perf_t *npp, struct timeval *tv1, struct timeval *tv2, uint64_t num_pkts)
{
	struct timeval tv_diff;
	uint64_t usecs;
	timersub(tv2, tv1, &tv_diff);
	usecs = tv_diff.tv_sec * 1000000ULL + tv_diff.tv_usec;
	OSAddAtomic64(usecs, &npp->np_total_usecs);
	OSAddAtomic64(num_pkts, &npp->np_total_pkts);
}

static void
update_bins(net_perf_t *npp, uint64_t bins)
{
	bzero(&npp->np_hist_bars, sizeof(npp->np_hist_bars));

	for (uint8_t i = 1, j = 0; i <= 64 && j < NET_PERF_BARS; i++) {
		if (bins & 0x1) {
			npp->np_hist_bars[j] = i;
			j++;
		}
		bins >>= 1;
	}
}

void
net_perf_initialize(net_perf_t *npp, uint64_t bins)
{
	bzero(npp, sizeof(net_perf_t));
	/* initialize np_hist_bars array */
	update_bins(npp, bins);
}

void
net_perf_histogram(net_perf_t *npp, uint64_t num_pkts)
{
	if (num_pkts <= npp->np_hist_bars[0]) {
		OSAddAtomic64(num_pkts, &npp->np_hist1);
	} else if (npp->np_hist_bars[0] < num_pkts && num_pkts <= npp->np_hist_bars[1]) {
		OSAddAtomic64(num_pkts, &npp->np_hist2);
	} else if (npp->np_hist_bars[1] < num_pkts && num_pkts <= npp->np_hist_bars[2]) {
		OSAddAtomic64(num_pkts, &npp->np_hist3);
	} else if (npp->np_hist_bars[2] < num_pkts && num_pkts <= npp->np_hist_bars[3]) {
		OSAddAtomic64(num_pkts, &npp->np_hist4);
	} else if (npp->np_hist_bars[3] < num_pkts) {
		OSAddAtomic64(num_pkts, &npp->np_hist5);
	}
}

boolean_t
net_perf_validate_bins(uint64_t bins)
{
	return NET_PERF_BARS == __builtin_popcountll(bins);
}
