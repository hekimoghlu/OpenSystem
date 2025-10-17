/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#define NETEM_MAX_BATCH_SIZE    32

__BEGIN_DECLS

struct kern_pbufpool;

typedef int (netem_output_func_t)(void *handle, pktsched_pkt_t *pkts,
    uint32_t n_pkts);

extern int netem_config(struct netem **ne, const char *__null_terminated name, struct ifnet *ifp,
    const struct if_netem_params *p, void *output_handle,
    netem_output_func_t *output_func, uint32_t output_max_batch_size);
extern void netem_get_params(struct netem *ne, struct if_netem_params *p);
extern void netem_destroy(struct netem *ne);
extern int netem_enqueue(struct netem *ne, classq_pkt_t *p, bool *pdrop);

__END_DECLS
