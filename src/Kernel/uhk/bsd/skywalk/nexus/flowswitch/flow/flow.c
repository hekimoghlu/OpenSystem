/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#include <skywalk/os_skywalk_private.h>
#include <skywalk/nexus/flowswitch/flow/flow_var.h>

#include <dev/random/randomdev.h>

#define SK_FO_ZONE_MAX                  256
#define SK_FO_ZONE_NAME                 "flow.owner"

unsigned int sk_fo_size;                /* size of zone element */
struct skmem_cache *sk_fo_cache;        /* cache for flow_owner */

#define SK_FE_ZONE_NAME                 "flow.entry"

unsigned int sk_fe_size;                /* size of zone element */
struct skmem_cache *sk_fe_cache;        /* cache for flow_entry */

#define SK_FAB_ZONE_NAME                "flow.adv.bmap"

unsigned int sk_fab_size;               /* size of zone element */
struct skmem_cache *sk_fab_cache;       /* cache for flow advisory bitmap */

static int __flow_inited = 0;
uint32_t flow_seed;

#define SKMEM_TAG_FLOW_DEMUX "com.apple.skywalk.fsw.flow_demux"
SKMEM_TAG_DEFINE(skmem_tag_flow_demux, SKMEM_TAG_FLOW_DEMUX);

int
flow_init(void)
{
	SK_LOCK_ASSERT_HELD();
	ASSERT(!__flow_inited);

	do {
		read_random(&flow_seed, sizeof(flow_seed));
	} while (flow_seed == 0);

	sk_fo_size = sizeof(struct flow_owner);
	if (sk_fo_cache == NULL) {
		sk_fo_cache = skmem_cache_create(SK_FO_ZONE_NAME, sk_fo_size,
		    sizeof(uint64_t), NULL, NULL, NULL, NULL, NULL, 0);
		if (sk_fo_cache == NULL) {
			panic("%s: skmem_cache create failed (%s)", __func__,
			    SK_FO_ZONE_NAME);
			/* NOTREACHED */
			__builtin_unreachable();
		}
	}

	sk_fe_size = sizeof(struct flow_entry);
	if (sk_fe_cache == NULL) {
		/* request for 16-bytes alignment (due to fe_key) */
		sk_fe_cache = skmem_cache_create(SK_FE_ZONE_NAME, sk_fe_size,
		    16, NULL, NULL, NULL, NULL, NULL, 0);
		if (sk_fe_cache == NULL) {
			panic("%s: skmem_cache create failed (%s)", __func__,
			    SK_FE_ZONE_NAME);
			/* NOTREACHED */
			__builtin_unreachable();
		}
	}

	/* these are initialized in skywalk_init() */
	VERIFY(sk_max_flows > 0 && sk_max_flows <= NX_FLOWADV_MAX);
	VERIFY(sk_fadv_nchunks != 0);
	_CASSERT(sizeof(*((struct flow_owner *)0)->fo_flowadv_bmap) ==
	    sizeof(bitmap_t));

	sk_fab_size = (sk_fadv_nchunks * sizeof(bitmap_t));
	if (sk_fab_cache == NULL) {
		sk_fab_cache = skmem_cache_create(SK_FAB_ZONE_NAME, sk_fab_size,
		    sizeof(uint64_t), NULL, NULL, NULL, NULL, NULL, 0);
		if (sk_fab_cache == NULL) {
			panic("%s: skmem_cache create failed (%s)", __func__,
			    SK_FAB_ZONE_NAME);
			/* NOTREACHED */
			__builtin_unreachable();
		}
	}

	flow_route_init();
	flow_stats_init();

	__flow_inited = 1;

	return 0;
}

void
flow_fini(void)
{
	if (__flow_inited) {
		flow_stats_fini();

		flow_route_fini();

		if (sk_fo_cache != NULL) {
			skmem_cache_destroy(sk_fo_cache);
			sk_fo_cache = NULL;
		}
		if (sk_fe_cache != NULL) {
			skmem_cache_destroy(sk_fe_cache);
			sk_fe_cache = NULL;
		}
		if (sk_fab_cache != NULL) {
			skmem_cache_destroy(sk_fab_cache);
			sk_fab_cache = NULL;
		}
		__flow_inited = 0;
	}
}
