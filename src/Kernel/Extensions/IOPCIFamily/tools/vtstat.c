/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <IOKit/IOKitLib.h>


typedef uint32_t ppnum_t;
#define arrayCount(x)	(sizeof(x) / sizeof(x[0]))

struct vtd_space_stats
{
    ppnum_t vsize;
    ppnum_t tables;
    ppnum_t bused;
    ppnum_t rused;
    ppnum_t largest_paging;
    ppnum_t largest_32b;
    ppnum_t inserts;
    ppnum_t max_inval[2];
    ppnum_t breakups;
    ppnum_t merges;
    ppnum_t allocs[64];
	ppnum_t bcounts[20];
};
typedef struct vtd_space_stats vtd_space_stats_t;

int main(int argc, char * argv[])
{
    io_service_t		vtd;
    CFDataRef			statsData;
    vtd_space_stats_t *	stats;
    uint32_t			idx;
    uint64_t            totalAllocs;

    vtd = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleVTD"));
    assert(vtd);
	statsData = IORegistryEntryCreateCFProperty(vtd, CFSTR("stats"),
								kCFAllocatorDefault, kNilOptions);
    assert(statsData);

	stats = (vtd_space_stats_t *) CFDataGetBytePtr(statsData);

	printf("vsize          0x%x\n", stats->vsize);
	printf("tables         0x%x\n", stats->tables);
	printf("bused          0x%x\n", stats->bused);
	printf("rused          0x%x\n", stats->rused);
	printf("largest_paging 0x%x\n", stats->largest_paging);
	printf("largest_32b    0x%x\n", stats->largest_32b);
	printf("max_binval     0x%x\n", stats->max_inval[0]);
	printf("max_rinval     0x%x\n", stats->max_inval[1]);
	printf("inserts        0x%x\n", stats->inserts);

	for (totalAllocs = 0, idx = 0; idx < arrayCount(stats->allocs); idx++)
	{
		printf("allocs[%2d]    0x%x\n", idx, stats->allocs[idx]);
		totalAllocs += stats->allocs[idx];
	}

	printf("breakups       0x%x(%.3f)\n", stats->breakups, stats->breakups * 100.0 / totalAllocs);
	printf("merges         0x%x(%.3f)\n", stats->breakups, stats->breakups * 100.0 / totalAllocs);

	for (idx = 0; idx < arrayCount(stats->bcounts); idx++)	printf("bcounts[%2d]    0x%x\n", idx, stats->bcounts[idx]);

	exit(0);
}
