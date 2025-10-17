/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#ifndef _SKYWALK_PACKET_PACKETQUEUE_H_
#define _SKYWALK_PACKET_PACKETQUEUE_H_

#ifdef BSD_KERNEL_PRIVATE

/*
 * Simple __kern_packet queueing system
 * This is basically a SIMPLEQ adapted to skywalk __kern_packet use.
 */
#define KPKTQ_HEAD(name)                                        \
struct name {                                                   \
	struct __kern_packet *kq_first; /* first packet */              \
	struct __kern_packet **kq_last; /* addr of last next packet */  \
	uint32_t kq_len; /* number of packets in queue */       \
}

#define KPKTQ_INIT(q)           do {                            \
	KPKTQ_FIRST(q) = NULL;                                  \
	(q)->kq_last = &KPKTQ_FIRST(q);                         \
	(q)->kq_len = 0;                                        \
} while (0)

#define KPKTQ_FINI(q)           do {                            \
	ASSERT(KPKTQ_EMPTY(q));                                 \
	ASSERT(KPKTQ_LEN(q) == 0);                              \
	KPKTQ_INIT(q);                                          \
} while (0)

#define KPKTQ_DISPOSE(q)        KPKTQ_INIT(q)

#define KPKTQ_CONCAT(q1, q2)    do {                            \
	if (!KPKTQ_EMPTY(q2)) {                                 \
	        *(q1)->kq_last = KPKTQ_FIRST(q2);               \
	        (q1)->kq_last = (q2)->kq_last;                  \
	        (q1)->kq_len += (q2)->kq_len;                   \
	        KPKTQ_DISPOSE((q2));                            \
	}                                                       \
} while (0)

#define KPKTQ_PREPEND(q, p)     do {                            \
	if ((KPKTQ_NEXT(p) = KPKTQ_FIRST(q)) == NULL) {         \
	        ASSERT((q)->kq_len == 0);                       \
	        (q)->kq_last = &KPKTQ_NEXT(p);                 \
	}                                                       \
	KPKTQ_FIRST(q) = (p);                                   \
	(q)->kq_len++;                                          \
} while (0)

#define KPKTQ_ENQUEUE(q, p)     do {                            \
	ASSERT(KPKTQ_NEXT(p) == NULL);                          \
	*(q)->kq_last = (p);                                    \
	(q)->kq_last = &KPKTQ_NEXT(p);                          \
	(q)->kq_len++;                                          \
} while (0)

#define KPKTQ_ENQUEUE_MULTI(q, p, n, c)    do {                 \
	KPKTQ_NEXT(n) = NULL;                                   \
	*(q)->kq_last = (p);                                    \
	(q)->kq_last = &KPKTQ_NEXT(n);                          \
	(q)->kq_len += c;                                       \
} while (0)

#define KPKTQ_ENQUEUE_LIST(q, p)           do {                 \
	uint32_t _c = 1;                                        \
	struct __kern_packet *_n = (p);                         \
	while (__improbable(KPKTQ_NEXT(_n) != NULL)) {          \
	        _c++;                                           \
	        _n = KPKTQ_NEXT(_n);                            \
	}                                                       \
	KPKTQ_ENQUEUE_MULTI(q, p, _n, _c);                      \
} while (0)

#define KPKTQ_DEQUEUE(q, p)     do {                            \
	if (((p) = KPKTQ_FIRST(q)) != NULL) {                   \
	        (q)->kq_len--;                                  \
	        if ((KPKTQ_FIRST(q) = KPKTQ_NEXT(p)) == NULL) { \
	                ASSERT((q)->kq_len == 0);               \
	                (q)->kq_last = &KPKTQ_FIRST(q);         \
	        } else {                                        \
	                KPKTQ_NEXT(p) = NULL;                   \
	        }                                               \
	}                                                       \
} while (0)

#define KPKTQ_REMOVE(q, p)      do {                            \
	if (KPKTQ_FIRST(q) == (p)) {                            \
	        KPKTQ_DEQUEUE(q, p);                            \
	} else {                                                \
	        struct __kern_packet *_p = KPKTQ_FIRST(q);      \
	        while (KPKTQ_NEXT(_p) != (p))                   \
	                _p = KPKTQ_NEXT(_p);                    \
	        if ((KPKTQ_NEXT(_p) =                           \
	            KPKTQ_NEXT(KPKTQ_NEXT(_p))) == NULL) {      \
	                (q)->kq_last = &KPKTQ_NEXT(_p);         \
	        }                                               \
	        (q)->kq_len--;                                  \
	        KPKTQ_NEXT(p) = NULL;                           \
	}                                                       \
} while (0)

#define KPKTQ_FOREACH(p, q)                                     \
	for ((p) = KPKTQ_FIRST(q);                              \
	    (p);                                                \
	    (p) = KPKTQ_NEXT(p))

#define KPKTQ_FOREACH_SAFE(p, q, tvar)                          \
	for ((p) = KPKTQ_FIRST(q);                              \
	    (p) && ((tvar) = KPKTQ_NEXT(p), 1);                 \
	    (p) = (tvar))

#define KPKTQ_EMPTY(q)          ((q)->kq_first == NULL)
#define KPKTQ_FIRST(q)          ((q)->kq_first)
#define KPKTQ_NEXT(p)           ((p)->pkt_nextpkt)
#define KPKTQ_LEN(q)            ((q)->kq_len)

/*
 * kq_last is initialized to point to kq_first, so check if they're
 * equal and return NULL when the list is empty.  Otherwise, we need
 * to subtract the offset of KPKTQ_NEXT (i.e. pkt_nextpkt field) to get
 * to the base packet address to return to caller.
 */
#define KPKTQ_LAST(head)                                        \
	(((head)->kq_last == &KPKTQ_FIRST(head)) ? NULL :       \
	__container_of((head)->kq_last, struct __kern_packet, pkt_nextpkt))

/*
 * struct pktq serves as basic common batching data structure using KPKTQ.
 * Elementary types of batch data structure, e.g. packet array, should be named
 * as pkts.
 * For instance:
 * rx_dequeue_pktq(struct pktq *pktq);
 * rx_dequeue_pkts(struct __kern_packet *pkts[], uint32_t n_pkts);
 */
KPKTQ_HEAD(pktq);

#endif /* BSD_KERNEL_PRIVATE */
#endif /* !_SKYWALK_PACKET_PACKETQUEUE_H_ */
