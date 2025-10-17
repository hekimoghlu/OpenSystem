/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
 * Copyright (c) 1980, 1986, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *	@(#)raw_cb.c	8.1 (Berkeley) 6/10/93
 */

#include <sys/param.h>
#include <sys/malloc.h>
#include <sys/socket.h>
#include <sys/socketvar.h>
#include <sys/domain.h>
#include <sys/protosw.h>
#include <kern/locks.h>

#include <net/raw_cb.h>

/*
 * Routines to manage the raw protocol control blocks.
 *
 * TODO:
 *	hash lookups by protocol family/protocol + address family
 *	take care of unique address problems per AF?
 *	redo address binding to allow wildcards
 */

struct rawcb_list_head rawcb_list = LIST_HEAD_INITIALIZER(rawcb_list);

static uint32_t raw_sendspace = RAWSNDQ;
static uint32_t raw_recvspace = RAWRCVQ;
extern lck_mtx_t        raw_mtx;       /*### global raw cb mutex for now */

/*
 * Allocate a control block and a nominal amount
 * of buffer space for the socket.
 */
int
raw_attach(struct socket *so, int proto)
{
	struct rawcb *rp = sotorawcb(so);
	int error;

	/*
	 * It is assumed that raw_attach is called
	 * after space has been allocated for the
	 * rawcb.
	 */
	if (rp == 0) {
		return ENOBUFS;
	}
	error = soreserve(so, raw_sendspace, raw_recvspace);
	if (error) {
		return error;
	}
	rp->rcb_socket = so;
	rp->rcb_proto.sp_family = (uint16_t)SOCK_DOM(so);
	rp->rcb_proto.sp_protocol = (uint16_t)proto;
	lck_mtx_lock(&raw_mtx);
	LIST_INSERT_HEAD(&rawcb_list, rp, list);
	lck_mtx_unlock(&raw_mtx);
	return 0;
}

/*
 * Detach the raw connection block and discard
 * socket resources.
 */
void
raw_detach_nofree(struct rawcb *rp)
{
	struct socket *so = rp->rcb_socket;

	so->so_pcb = 0;
	so->so_flags |= SOF_PCBCLEARING;
	sofree(so);
	if (!lck_mtx_try_lock(&raw_mtx)) {
		socket_unlock(so, 0);
		lck_mtx_lock(&raw_mtx);
		socket_lock(so, 0);
	}
	LIST_REMOVE(rp, list);
	lck_mtx_unlock(&raw_mtx);
	rp->rcb_socket = NULL;
}

/*
 * Disconnect and possibly release resources.
 */
void
raw_disconnect(struct rawcb *rp)
{
	struct socket *so = rp->rcb_socket;

	/*
	 * A multipath subflow socket would have its SS_NOFDREF set by default,
	 * so check for SOF_MP_SUBFLOW socket flag before detaching the PCB;
	 * when the socket is closed for real, SOF_MP_SUBFLOW would be cleared.
	 */
	if (!(so->so_flags & SOF_MP_SUBFLOW) && (so->so_state & SS_NOFDREF)) {
		raw_detach_nofree(rp);
		kfree_type(struct rawcb, rp);
	}
}
