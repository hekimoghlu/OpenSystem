/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include <stdint.h>
#include <sys/cdefs.h> /* prerequisite */
#include <sys/types.h>
#include <sys/param.h>
#include <sys/systm.h>
#include <skywalk/nexus/nexus_pktq.h>

static void __nx_pktq_init(struct nx_pktq *q, uint32_t, lck_grp_t *lck_grp);

__attribute__((always_inline))
static inline void
__nx_pktq_init(struct nx_pktq *q, uint32_t lim, lck_grp_t *lck_grp)
{
	bzero(q, sizeof(*q));
	_qinit(&q->nx_pktq_q, Q_DROPTAIL, lim, QP_PACKET);
	q->nx_pktq_grp = lck_grp;
}

void
nx_pktq_safe_init(struct __kern_channel_ring *kr, struct nx_pktq *q,
    uint32_t lim, lck_grp_t *lck_grp, lck_attr_t *lck_attr)
{
	q->nx_pktq_kring = kr;
	__nx_pktq_init(q, lim, lck_grp);
	lck_mtx_init(&q->nx_pktq_lock, lck_grp, lck_attr);
}

void
nx_pktq_init(struct nx_pktq *q, uint32_t lim)
{
	__nx_pktq_init(q, lim, NULL);
}

void
nx_pktq_concat(struct nx_pktq *q1, struct nx_pktq *q2)
{
	uint32_t qlen;
	uint64_t qsize;
	classq_pkt_t first = CLASSQ_PKT_INITIALIZER(first);
	classq_pkt_t last = CLASSQ_PKT_INITIALIZER(last);

	/* caller is responsible for locking */
	if (!nx_pktq_empty(q2)) {
		_getq_all(&q2->nx_pktq_q, &first, &last, &qlen, &qsize);
		ASSERT(first.cp_kpkt != NULL && last.cp_kpkt != NULL);
		_addq_multi(&q1->nx_pktq_q, &first, &last, qlen, qsize);
		ASSERT(nx_pktq_empty(q2));
	}
}

boolean_t
nx_pktq_empty(struct nx_pktq *q)
{
	return qempty(&q->nx_pktq_q) && qhead(&q->nx_pktq_q) == NULL;
}

void
nx_pktq_purge(struct nx_pktq *q)
{
	_flushq(&q->nx_pktq_q);
}

void
nx_pktq_safe_purge(struct nx_pktq *q)
{
	nx_pktq_lock(q);
	_flushq(&q->nx_pktq_q);
	nx_pktq_unlock(q);
}

void
nx_pktq_safe_destroy(struct nx_pktq *q)
{
	VERIFY(nx_pktq_empty(q));
	lck_mtx_destroy(&q->nx_pktq_lock, q->nx_pktq_grp);
}

void
nx_pktq_destroy(struct nx_pktq *q)
{
	VERIFY(nx_pktq_empty(q));
}
