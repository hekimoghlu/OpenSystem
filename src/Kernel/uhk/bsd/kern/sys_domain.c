/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
 *	@(#)sys_domain.c       1.0 (6/1/2000)
 */

#include <sys/param.h>
#include <sys/protosw.h>
#include <sys/domain.h>
#include <sys/mcache.h>
#include <sys/sys_domain.h>
#include <sys/sysctl.h>

struct domain *systemdomain = NULL;

/* domain init function */
static void systemdomain_init(struct domain *);

struct domain systemdomain_s = {
	.dom_family =           PF_SYSTEM,
	.dom_name =             "system",
	.dom_init =             systemdomain_init,
};

SYSCTL_NODE(_net, PF_SYSTEM, systm,
    CTLFLAG_RW | CTLFLAG_LOCKED, 0, "System domain");


static void
systemdomain_init(struct domain *dp)
{
	VERIFY(!(dp->dom_flags & DOM_INITIALIZED));
	VERIFY(systemdomain == NULL);

	systemdomain = dp;

	/* add system domain built in protocol initializers here */
	kern_event_init(dp);
	kern_control_init(dp);
}
