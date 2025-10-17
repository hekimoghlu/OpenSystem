/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
/*	$KAME: if_gif.h,v 1.7 2000/02/22 14:01:46 itojun Exp $	*/

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

/*
 * if_gif.h
 */

#ifndef _NET_IF_GIF_H_
#define _NET_IF_GIF_H_
#include <sys/appleapiopts.h>

#include <netinet/in.h>
/* xxx sigh, why route have struct route instead of pointer? */

#ifdef BSD_KERNEL_PRIVATE

extern void gif_init(void);

struct encaptab;

struct gif_softc {
	ifnet_t                 gif_if;    /* pointer back to the interface */
	struct sockaddr *gif_psrc; /* Physical src addr */
	struct sockaddr *gif_pdst; /* Physical dst addr */
#ifdef __APPLE__
	protocol_family_t gif_proto; /* dlil protocol attached */
#endif
	union {
		struct route  gifscr_ro;    /* xxx */
		struct route_in6 gifscr_ro6; /* xxx */
	} gifsc_gifscr;
	int             gif_flags;
#define IFGIF_DETACHING 0x1
	int             gif_called;
	const struct encaptab *encap_cookie4;
	const struct encaptab *encap_cookie6;
	TAILQ_ENTRY(gif_softc) gif_link; /* all gif's are linked */
	bpf_tap_mode    tap_mode;
	bpf_packet_func tap_callback;
	char    gif_ifname[IFNAMSIZ];
	decl_lck_mtx_data(, gif_lock);  /* lock for gif softc structure */
};

#define GIF_LOCK(_sc)           lck_mtx_lock(&(_sc)->gif_lock)
#define GIF_UNLOCK(_sc)         lck_mtx_unlock(&(_sc)->gif_lock)
#define GIF_LOCK_ASSERT(_sc)    LCK_MTX_ASSERT(&(_sc)->gif_lock,        \
    LCK_MTX_ASSERT_OWNED)

#define gif_ro gifsc_gifscr.gifscr_ro
#define gif_ro6 gifsc_gifscr.gifscr_ro6

#endif /* BSD_KERNEL_PRIVATE */

#define GIF_MTU         (1280)  /* Default MTU */
#define GIF_MTU_MIN     (1280)  /* Minimum MTU */
#define GIF_MTU_MAX     (8192)  /* Maximum MTU */

#endif /* _NET_IF_GIF_H_ */
