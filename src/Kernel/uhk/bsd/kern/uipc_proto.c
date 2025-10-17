/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
 *//*-
 * Copyright (c) 1982, 1986, 1993
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
 *	@(#)uipc_proto.c	8.2 (Berkeley) 2/14/95
 */

#include <sys/param.h>
#include <sys/socket.h>
#include <sys/protosw.h>
#include <sys/domain.h>
#include <sys/mbuf.h>
#include <sys/mcache.h>
#include <sys/un.h>
#include <net/raw_cb.h>
#include <sys/sysctl.h>

/*
 * Definitions of protocols supported in the UNIX domain.
 */
struct domain *localdomain = NULL;
static void pre_unp_init(struct domain *);

extern struct domain localdomain_s;

static struct protosw localsw[] = {
	{
		.pr_type =      SOCK_STREAM,
		.pr_flags =     PR_CONNREQUIRED | PR_WANTRCVD | PR_RIGHTS | PR_PCBLOCK,
		.pr_ctloutput = uipc_ctloutput,
		.pr_usrreqs =   &uipc_usrreqs,
		.pr_lock =      unp_lock,
		.pr_unlock =    unp_unlock,
		.pr_getlock =   unp_getlock
	},
	{
		.pr_type =      SOCK_DGRAM,
		.pr_flags =     PR_ATOMIC | PR_WANTRCVD | PR_ADDR | PR_RIGHTS,
		.pr_ctloutput = uipc_ctloutput,
		.pr_usrreqs =   &uipc_usrreqs,
		.pr_lock =      unp_lock,
		.pr_unlock =    unp_unlock,
		.pr_getlock =   unp_getlock
	},
	{
		.pr_ctlinput =  raw_ctlinput,
		.pr_usrreqs =   &raw_usrreqs,
	},
};

static int local_proto_count = (sizeof(localsw) / sizeof(struct protosw));

static void
pre_unp_init(struct domain *dp)
{
	struct protosw *pr;
	int i;

	VERIFY(!(dp->dom_flags & DOM_INITIALIZED));
	VERIFY(localdomain == NULL);

	localdomain = dp;

	for (i = 0, pr = &localsw[0]; i < local_proto_count; i++, pr++) {
		net_add_proto(pr, dp, 1);
	}

	unp_init();
}

struct domain localdomain_s = {
	.dom_family =           PF_LOCAL,
	.dom_name =             "unix",
	.dom_init =             pre_unp_init,
	.dom_externalize =      unp_externalize,
	.dom_dispose =          unp_dispose,
};

SYSCTL_NODE(_net, PF_LOCAL, local, CTLFLAG_RW | CTLFLAG_LOCKED,
    NULL, "Local domain");
SYSCTL_NODE(_net_local, SOCK_STREAM, stream, CTLFLAG_RW | CTLFLAG_LOCKED,
    NULL, "SOCK_STREAM");
SYSCTL_NODE(_net_local, SOCK_DGRAM, dgram, CTLFLAG_RW | CTLFLAG_LOCKED,
    NULL, "SOCK_DGRAM");

