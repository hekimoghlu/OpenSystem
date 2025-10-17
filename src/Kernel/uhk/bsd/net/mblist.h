/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
/*
 * mblist.h
 */

#ifndef _NET_MBLIST_H
#define _NET_MBLIST_H

#include <sys/kpi_mbuf.h>
#include <stdint.h>

/*
 * Type: mblist
 * Purpose:
 *   Simple type to store head, tail pointers for a list of mbuf packets.
 */
typedef struct {
	mbuf_t          head;
	mbuf_t          tail;
	uint32_t        bytes;
	uint32_t        count;
} mblist, * mblist_t;

static inline void
mblist_init(mblist_t list)
{
	bzero(list, sizeof(*list));
}

static inline void
mblist_append(mblist_t list, mbuf_t m)
{
	if (list->head == NULL) {
		list->head = m;
	} else {
		list->tail->m_nextpkt = m;
	}
	list->tail = m;
	list->count++;
	list->bytes += mbuf_pkthdr_len(m);
}

static inline void
mblist_append_list(mblist_t list, mblist append)
{
	VERIFY(append.head != NULL);
	if (list->head == NULL) {
		*list = append;
	} else {
		VERIFY(list->tail != NULL);
		list->tail->m_nextpkt = append.head;
		list->tail = append.tail;
		list->count += append.count;
		list->bytes += append.bytes;
	}
}

#endif /* _NET_MBLIST_H */
