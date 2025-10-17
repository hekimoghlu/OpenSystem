/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#ifndef lint
#if 0
static char sccsid[] = "@(#)print.c	8.1 (Berkeley) 6/4/93";
#endif
__attribute__((__used__))
static const char rcsid[] =
  "$FreeBSD$";
#endif /* not lint */

/* debug print routines */

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <protocols/talkd.h>
#include <stdio.h>
#include <syslog.h>

#include "extern.h"

static	const char *types[] =
    { "leave_invite", "look_up", "delete", "announce" };
#define	NTYPES	(sizeof (types) / sizeof (types[0]))
static	const char *answers[] =
    { "success", "not_here", "failed", "machine_unknown", "permission_denied",
      "unknown_request", "badversion", "badaddr", "badctladdr" };
#define	NANSWERS	(sizeof (answers) / sizeof (answers[0]))

void
print_request(const char *cp, CTL_MSG *mp)
{
	const char *tp;
	char tbuf[80];

	if (mp->type > NTYPES) {
		(void)snprintf(tbuf, sizeof(tbuf), "type %d", mp->type);
		tp = tbuf;
	} else
		tp = types[mp->type];
	syslog(LOG_DEBUG, "%s: %s: id %lu, l_user %s, r_user %s, r_tty %s",
	    cp, tp, (long)mp->id_num, mp->l_name, mp->r_name, mp->r_tty);
}

void
print_response(const char *cp, CTL_RESPONSE *rp)
{
	const char *tp, *ap;
	char tbuf[80], abuf[80];

	if (rp->type > NTYPES) {
		(void)snprintf(tbuf, sizeof(tbuf), "type %d", rp->type);
		tp = tbuf;
	} else
		tp = types[rp->type];
	if (rp->answer > NANSWERS) {
		(void)snprintf(abuf, sizeof(abuf), "answer %d", rp->answer);
		ap = abuf;
	} else
		ap = answers[rp->answer];
	syslog(LOG_DEBUG, "%s: %s: %s, id %d", cp, tp, ap, ntohl(rp->id_num));
}
