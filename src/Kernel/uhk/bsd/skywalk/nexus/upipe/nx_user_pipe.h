/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
 * Copyright (C) 2011-2014 Matteo Landi, Luigi Rizzo. All rights reserved.
 * Copyright (C) 2013-2014 Universita` di Pisa. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _SKYWALK_NEXUS_USER_PIPE_H_
#define _SKYWALK_NEXUS_USER_PIPE_H_

#include <skywalk/os_skywalk_private.h>

#if CONFIG_NEXUS_USER_PIPE
#define NX_UPIPE_MAXPIPES       64      /* max number of pipes per adapter */

struct nexus_upipe_adapter {
	/*
	 * This is an overlay structure on nexus_adapter;
	 * make sure it contains 'up' as the first member.
	 */
	struct nexus_adapter pna_up;

	uint32_t        pna_id;                 /* pipe identifier */
	ch_endpoint_t   pna_role;               /* master or slave */

	struct nexus_adapter *pna_parent; /* adapter that owns the memory */
	struct nexus_upipe_adapter *pna_peer; /* the other end of the pipe */
	boolean_t pna_peer_ref; /* 1 iff we are holding a ref to the peer */

	uint32_t pna_parent_slot; /* index in the parent pipe array */
};

/*
 * nx_upipe is a descriptor for a user pipe nexus instance.
 */
struct nx_upipe {
	struct nexus_adapter    *nup_pna;
	uint32_t                nup_pna_users;
	struct nxbind           *nup_cli_nxb;
	struct nxbind           *nup_srv_nxb;
};

#define NX_UPIPE_PRIVATE(_nx) ((struct nx_upipe *)(_nx)->nx_arg)

#define NEXUS_PROVIDER_USER_PIPE "com.apple.nexus.upipe"

extern struct nxdom nx_upipe_dom_s;

__BEGIN_DECLS
extern void nx_upipe_na_dealloc(struct nexus_adapter *);
extern int nx_upipe_na_find(struct kern_nexus *, struct kern_channel *,
    struct chreq *, struct nxbind *, struct proc *, struct nexus_adapter **,
    boolean_t);
__END_DECLS
#else /* !CONFIG_NEXUS_USER_PIPE */
#define NM_MAXPIPES     0
#endif /* !CONFIG_NEXUS_USER_PIPE */
#endif /* _SKYWALK_NEXUS_USER_PIPE_H_ */
