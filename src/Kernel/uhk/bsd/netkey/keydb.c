/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
/*	$KAME: keydb.c,v 1.61 2000/03/25 07:24:13 sumikawa Exp $	*/

/*
 * Copyright (C) 1995, 1996, 1997, and 1998 WIDE Project.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <sys/systm.h>
#include <sys/kernel.h>
#include <sys/malloc.h>
#include <sys/errno.h>
#include <sys/queue.h>

#include <net/if.h>
#include <net/route.h>

#include <netinet/in.h>

#include <net/pfkeyv2.h>
#include <netkey/key.h>
#include <netkey/keydb.h>
#include <netinet6/ipsec.h>

#include <net/net_osdep.h>

// static void keydb_delsecasvar(struct secasvar *); // not used

/*
 * secpolicy management
 */
struct secpolicy *
keydb_newsecpolicy(void)
{
	LCK_MTX_ASSERT(sadb_mutex, LCK_MTX_ASSERT_NOTOWNED);

	return kalloc_type(struct secpolicy, Z_WAITOK | Z_ZERO);
}

void
keydb_delsecpolicy(struct secpolicy *p)
{
	kfree_type(struct secpolicy, p);
}

/*
 * secashead management
 */
struct secashead *
keydb_newsecashead(void)
{
	struct secashead *p;

	LCK_MTX_ASSERT(sadb_mutex, LCK_MTX_ASSERT_OWNED);

	p = kalloc_type(struct secashead, Z_NOWAIT | Z_ZERO);
	if (!p) {
		lck_mtx_unlock(sadb_mutex);
		p = kalloc_type(struct secashead, Z_WAITOK | Z_ZERO | Z_NOFAIL);
		lck_mtx_lock(sadb_mutex);
	}
	for (size_t i = 0; i < ARRAY_COUNT(p->savtree); i++) {
		LIST_INIT(&p->savtree[i]);
	}
	return p;
}

/*
 * secreplay management
 */
struct secreplay *
keydb_newsecreplay(u_int8_t wsize)
{
	struct secreplay *p;
	caddr_t tmp_bitmap = NULL;

	LCK_MTX_ASSERT(sadb_mutex, LCK_MTX_ASSERT_OWNED);

	p = kalloc_type(struct secreplay, Z_NOWAIT | Z_ZERO);
	if (!p) {
		lck_mtx_unlock(sadb_mutex);
		p = kalloc_type(struct secreplay, Z_WAITOK | Z_ZERO | Z_NOFAIL);
		lck_mtx_lock(sadb_mutex);
	}

	if (wsize != 0) {
		tmp_bitmap = (caddr_t)kalloc_data(wsize, Z_NOWAIT | Z_ZERO);
		if (!tmp_bitmap) {
			lck_mtx_unlock(sadb_mutex);
			tmp_bitmap = (caddr_t)kalloc_data(wsize, Z_WAITOK | Z_ZERO | Z_NOFAIL);
			lck_mtx_lock(sadb_mutex);
		}

		p->bitmap = tmp_bitmap;
		p->wsize = wsize;
	}
	return p;
}

void
keydb_delsecreplay(struct secreplay *p)
{
	if (p->bitmap) {
		kfree_data_sized_by(p->bitmap, p->wsize);
	}
	kfree_type(struct secreplay, p);
}
