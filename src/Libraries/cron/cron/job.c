/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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
#if !defined(lint) && !defined(LINT)
static const char rcsid[] =
  "$FreeBSD: src/usr.sbin/cron/cron/job.c,v 1.6 1999/08/28 01:15:50 peter Exp $";
#endif


#include "cron.h"

#ifdef __APPLE__
#include <btm.h>
#include <os/feature_private.h>
#endif

typedef	struct _job {
	struct _job	*next;
	entry		*e;
	user		*u;
} job;


static job	*jhead = NULL, *jtail = NULL;


void
job_add(e, u)
	register entry *e;
	register user *u;
{
	register job *j;

	/* if already on queue, keep going */
	for (j=jhead; j; j=j->next)
		if (j->e == e && j->u == u) { return; }

	/* build a job queue element */
	if ((j = (job*)malloc(sizeof(job))) == NULL)
		return;
	j->next = (job*) NULL;
	j->e = e;
	j->u = u;

	/* add it to the tail */
	if (!jhead) { jhead=j; }
	else { jtail->next=j; }
	jtail = j;
}


int
job_runqueue()
{
	register job	*j, *jn;
	register int	run = 0;

	for (j=jhead; j; j=jn) {
#ifdef __APPLE__
		if (os_feature_enabled(cronBTMToggle, cronBTMCheck)) {
			bool cron_enabled = FALSE;
			btm_error_code_t error = btm_get_enablement_status_for_subsystem_and_uid(btm_subsystem_cron, BTMGlobalDataUID, &cron_enabled);

			if (error != btm_error_none) {
				Debug(DMISC, ("Error contacting BTM to check enablement state: %d", error));
			}

			if (cron_enabled) {
				do_command(j->e, j->u);
			}
		} else {
			do_command(j->e, j->u);
		}
#else
		do_command(j->e, j->u);
#endif
		jn = j->next;
		free(j);
		run++;
	}
	jhead = jtail = NULL;
	return run;
}
