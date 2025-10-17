/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
static char sccsid[] = "@(#)process.c	8.2 (Berkeley) 11/16/93";
#endif
__attribute__((__used__))
static const char rcsid[] =
  "$FreeBSD$";
#endif /* not lint */

/*
 * process.c handles the requests, which can be of three types:
 *	ANNOUNCE - announce to a user that a talk is wanted
 *	LEAVE_INVITE - insert the request into the table
 *	LOOK_UP - look up to see if a request is waiting in
 *		  in the table for the local user
 *	DELETE - delete invitation
 */
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <protocols/talkd.h>
#include <ctype.h>
#include <err.h>
#include <netdb.h>
#include <paths.h>
#include <stdio.h>
#include <string.h>
#include <syslog.h>
#include <utmpx.h>

#include "extern.h"

void
process_request(CTL_MSG *mp, CTL_RESPONSE *rp)
{
	CTL_MSG *ptr;
	char *s;

	rp->vers = TALK_VERSION;
	rp->type = mp->type;
	rp->id_num = htonl(0);
	if (mp->vers != TALK_VERSION) {
		syslog(LOG_WARNING, "bad protocol version %d", mp->vers);
		rp->answer = BADVERSION;
		return;
	}
	mp->id_num = ntohl(mp->id_num);
	mp->addr.sa_family = ntohs(mp->addr.sa_family);
	if (mp->addr.sa_family != AF_INET) {
		syslog(LOG_WARNING, "bad address, family %d",
		    mp->addr.sa_family);
		rp->answer = BADADDR;
		return;
	}
	mp->ctl_addr.sa_family = ntohs(mp->ctl_addr.sa_family);
	if (mp->ctl_addr.sa_family != AF_INET) {
		syslog(LOG_WARNING, "bad control address, family %d",
		    mp->ctl_addr.sa_family);
		rp->answer = BADCTLADDR;
		return;
	}
	for (s = mp->l_name; *s; s++)
		if (!isprint(*s)) {
			syslog(LOG_NOTICE, "illegal user name. Aborting");
			rp->answer = FAILED;
			return;
		}
	mp->pid = ntohl(mp->pid);
	if (debug)
		print_request("process_request", mp);
	switch (mp->type) {

	case ANNOUNCE:
		do_announce(mp, rp);
		break;

	case LEAVE_INVITE:
		ptr = find_request(mp);
		if (ptr != (CTL_MSG *)0) {
			rp->id_num = htonl(ptr->id_num);
			rp->answer = SUCCESS;
		} else
			insert_table(mp, rp);
		break;

	case LOOK_UP:
		ptr = find_match(mp);
		if (ptr != (CTL_MSG *)0) {
			rp->id_num = htonl(ptr->id_num);
			rp->addr = ptr->addr;
			rp->addr.sa_family = htons(ptr->addr.sa_family);
			rp->answer = SUCCESS;
		} else
			rp->answer = NOT_HERE;
		break;

	case DELETE:
		rp->answer = delete_invite(mp->id_num);
		break;

	default:
		rp->answer = UNKNOWN_REQUEST;
		break;
	}
	if (debug)
		print_response("process_request", rp);
}

void
do_announce(CTL_MSG *mp, CTL_RESPONSE *rp)
{
	struct hostent *hp;
	CTL_MSG *ptr;
	int result;

	/* see if the user is logged */
	result = find_user(mp->r_name, mp->r_tty);
	if (result != SUCCESS) {
		rp->answer = result;
		return;
	}
#define	satosin(sa)	((struct sockaddr_in *)(void *)(sa))
	hp = gethostbyaddr(&satosin(&mp->ctl_addr)->sin_addr,
		sizeof (struct in_addr), AF_INET);
	if (hp == (struct hostent *)0) {
		rp->answer = MACHINE_UNKNOWN;
		return;
	}
	ptr = find_request(mp);
	if (ptr == (CTL_MSG *) 0) {
		insert_table(mp, rp);
		rp->answer = announce(mp, hp->h_name);
		return;
	}
	if (mp->id_num > ptr->id_num) {
		/*
		 * This is an explicit re-announce, so update the id_num
		 * field to avoid duplicates and re-announce the talk.
		 */
		ptr->id_num = new_id();
		rp->id_num = htonl(ptr->id_num);
		rp->answer = announce(mp, hp->h_name);
	} else {
		/* a duplicated request, so ignore it */
		rp->id_num = htonl(ptr->id_num);
		rp->answer = SUCCESS;
	}
}

/*
 * Search utmp for the local user
 */
int
find_user(const char *name, char *tty)
{
	struct utmpx *ut;
	int status;
	struct stat statb;
	time_t best = 0;
	char ftty[sizeof(_PATH_DEV) - 1 + sizeof(ut->ut_line)];

	setutxent();
	status = NOT_HERE;
	(void) strcpy(ftty, _PATH_DEV);
	while ((ut = getutxent()) != NULL)
		if (ut->ut_type == USER_PROCESS &&
		    strcmp(ut->ut_user, name) == 0) {
			if (*tty == '\0' || best != 0) {
				if (best == 0)
					status = PERMISSION_DENIED;
				/* no particular tty was requested */
				(void) strcpy(ftty + sizeof(_PATH_DEV) - 1,
				    ut->ut_line);
				if (stat(ftty, &statb) == 0) {
					if (!(statb.st_mode & 020))
						continue;
					if (statb.st_atime > best) {
						best = statb.st_atime;
						(void) strcpy(tty, ut->ut_line);
						status = SUCCESS;
						continue;
					}
				}
			}
			if (strcmp(ut->ut_line, tty) == 0) {
				status = SUCCESS;
				break;
			}
		}
	endutxent();
	return (status);
}
