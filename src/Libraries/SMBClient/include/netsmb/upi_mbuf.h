/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#ifndef _UPI_MBUF_H_
#define _UPI_MBUF_H_

#include <stdint.h>
/*
 * User land mbuf routine, never use in kernel code. See kpi_mbuf.h in the kernel 
 * for more details. This code was derived from the darwin source and tries to 
 * following the same kpis. Any differnece are kept in the routine themself
 * and never exposed to the calling process.
 *
 * See darwin ./bsd/sys/kpi_mbuf.h
 */
#ifndef KERNEL

enum {
	MBUF_TYPE_FREE		= 0,	/* should be on free list */
	MBUF_TYPE_DATA		= 1,	/* dynamic (data) allocation */
	MBUF_TYPE_HEADER	= 2,	/* packet header */
	MBUF_TYPE_SOCKET	= 3,	/* socket structure */
	MBUF_TYPE_PCB		= 4,	/* protocol control block */
	MBUF_TYPE_RTABLE	= 5,	/* routing tables */
	MBUF_TYPE_HTABLE	= 6,	/* IMP host tables */
	MBUF_TYPE_ATABLE	= 7,	/* address resolution tables */
	MBUF_TYPE_SONAME	= 8,	/* socket name */
	MBUF_TYPE_SOOPTS	= 10,	/* socket options */
	MBUF_TYPE_FTABLE	= 11,	/* fragment reassembly header */
	MBUF_TYPE_RIGHTS	= 12,	/* access rights */
	MBUF_TYPE_IFADDR	= 13,	/* interface address */
	MBUF_TYPE_CONTROL	= 14,	/* extra-data protocol message */
	MBUF_TYPE_OOBDATA	= 15	/* expedited data  */
};
typedef uint32_t mbuf_type_t;

/* Currently we only support MBUF_WAITOK */
enum {
	MBUF_WAITOK	= 0,	/* Ok to block to get memory */
	MBUF_DONTWAIT	= 1	/* Don't block, fail if blocking would be required */
};
typedef uint32_t mbuf_how_t;

struct smb_mbuf;
typedef	struct smb_mbuf* mbuf_t;

mbuf_t mbuf_free(mbuf_t mbuf);
void mbuf_freem(mbuf_t mbuf);
int mbuf_gethdr(mbuf_how_t how, mbuf_type_t type, mbuf_t *mbuf);
int mbuf_get(mbuf_how_t how, mbuf_type_t type, mbuf_t *mbuf);
int mbuf_getcluster(mbuf_how_t how, mbuf_type_t type, size_t size, mbuf_t *mbuf);
int mbuf_attachcluster(mbuf_how_t how, mbuf_type_t type,
					   mbuf_t *mbuf, void * extbuf, void (*extfree)(caddr_t , size_t, caddr_t),
					   size_t extsize, caddr_t extarg);
size_t mbuf_len(const mbuf_t mbuf);
size_t mbuf_maxlen(const mbuf_t mbuf);
void mbuf_setlen(mbuf_t mbuf, size_t len);
size_t mbuf_pkthdr_len(const mbuf_t mbuf);
void mbuf_pkthdr_setlen(mbuf_t mbuf, size_t len);
void mbuf_pkthdr_adjustlen(mbuf_t mbuf, int amount);
mbuf_t mbuf_next(const mbuf_t mbuf);
int mbuf_setnext(mbuf_t mbuf, mbuf_t next);
void * mbuf_data(const mbuf_t mbuf);
size_t mbuf_trailingspace(const mbuf_t mbuf);
int mbuf_copydata(const mbuf_t mbuf, size_t offset, size_t length, void *out_data);

#endif // #ifndef KERNEL
#endif // _UPI_MBUF_H_
