/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/socket.h>
#include <sys/domain.h>
#include <sys/protosw.h>
#include <sys/mcache.h>

#include <kern/locks.h>

#include <netinet/in.h>
#include <netinet/mptcp_var.h>

extern struct domain mpdomain_s;

static void mp_dinit(struct domain *);

static struct protosw mpsw = {
	.pr_type =              SOCK_STREAM,
	.pr_protocol =          IPPROTO_TCP,
	.pr_flags =             PR_CONNREQUIRED | PR_MULTICONN | PR_EVCONNINFO |
    PR_WANTRCVD | PR_PCBLOCK | PR_PROTOLOCK |
    PR_PRECONN_WRITE | PR_DATA_IDEMPOTENT,
	.pr_ctloutput =         mptcp_ctloutput,
	.pr_init =              mptcp_init,
	.pr_usrreqs =           &mptcp_usrreqs,
	.pr_lock =              mptcp_lock,
	.pr_unlock =            mptcp_unlock,
	.pr_getlock =           mptcp_getlock,
};

struct domain mpdomain_s = {
	.dom_family =           PF_MULTIPATH,
	.dom_flags =            DOM_REENTRANT,
	.dom_name =             "multipath",
	.dom_init =             mp_dinit,
};

/* Initialize the PF_MULTIPATH domain, and add in the pre-defined protos */
void
mp_dinit(struct domain *dp)
{
	VERIFY(!(dp->dom_flags & DOM_INITIALIZED));

	net_add_proto(&mpsw, dp, 1);
}
