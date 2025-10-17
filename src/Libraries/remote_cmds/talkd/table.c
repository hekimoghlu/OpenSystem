/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
static char sccsid[] = "@(#)table.c	8.1 (Berkeley) 6/4/93";
#endif
__attribute__((__used__))
static const char rcsid[] =
  "$FreeBSD$";
#endif /* not lint */

/*
 * Routines to handle insertion, deletion, etc on the table
 * of requests kept by the daemon. Nothing fancy here, linear
 * search on a double-linked list. A time is kept with each
 * entry so that overly old invitations can be eliminated.
 *
 * Consider this a mis-guided attempt at modularity
 */
#include <sys/param.h>
#include <sys/time.h>
#ifdef __APPLE__
#ifndef CLOCK_MONOTONIC_FAST
#define CLOCK_MONOTONIC_FAST CLOCK_MONOTONIC
#endif
#endif /* __APPLE__ */
#include <sys/socket.h>
#include <netinet/in.h>
#include <protocols/talkd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>

#include "extern.h"

#define MAX_ID 16000	/* << 2^15 so I don't have sign troubles */

#define NIL ((TABLE_ENTRY *)0)

static struct timespec ts;

typedef struct table_entry TABLE_ENTRY;

struct table_entry {
	CTL_MSG request;
	long	time;
	TABLE_ENTRY *next;
	TABLE_ENTRY *last;
};

static void delete(TABLE_ENTRY *);

static TABLE_ENTRY *table = NIL;

/*
 * Look in the table for an invitation that matches the current
 * request looking for an invitation
 */
CTL_MSG *
find_match(CTL_MSG *request)
{
	TABLE_ENTRY *ptr, *next;
	time_t current_time;

	clock_gettime(CLOCK_MONOTONIC_FAST, &ts);
	current_time = ts.tv_sec;
	if (debug)
		print_request("find_match", request);
	for (ptr = table; ptr != NIL; ptr = next) {
		next = ptr->next;
		if ((ptr->time - current_time) > MAX_LIFE) {
			/* the entry is too old */
			if (debug)
				print_request("deleting expired entry",
				    &ptr->request);
			delete(ptr);
			continue;
		}
		if (debug)
			print_request("", &ptr->request);
		if (strcmp(request->l_name, ptr->request.r_name) == 0 &&
		    strcmp(request->r_name, ptr->request.l_name) == 0 &&
		     ptr->request.type == LEAVE_INVITE)
			return (&ptr->request);
	}
	return ((CTL_MSG *)0);
}

/*
 * Look for an identical request, as opposed to a complimentary
 * one as find_match does
 */
CTL_MSG *
find_request(CTL_MSG *request)
{
	TABLE_ENTRY *ptr, *next;
	time_t current_time;

	clock_gettime(CLOCK_MONOTONIC_FAST, &ts);
	current_time = ts.tv_sec;
	/*
	 * See if this is a repeated message, and check for
	 * out of date entries in the table while we are it.
	 */
	if (debug)
		print_request("find_request", request);
	for (ptr = table; ptr != NIL; ptr = next) {
		next = ptr->next;
		if ((ptr->time - current_time) > MAX_LIFE) {
			/* the entry is too old */
			if (debug)
				print_request("deleting expired entry",
				    &ptr->request);
			delete(ptr);
			continue;
		}
		if (debug)
			print_request("", &ptr->request);
		if (strcmp(request->r_name, ptr->request.r_name) == 0 &&
		    strcmp(request->l_name, ptr->request.l_name) == 0 &&
		    request->type == ptr->request.type &&
		    request->pid == ptr->request.pid) {
			/* update the time if we 'touch' it */
			ptr->time = current_time;
			return (&ptr->request);
		}
	}
	return ((CTL_MSG *)0);
}

void
insert_table(CTL_MSG *request, CTL_RESPONSE *response)
{
	TABLE_ENTRY *ptr;
	time_t current_time;

	clock_gettime(CLOCK_MONOTONIC_FAST, &ts);
	current_time = ts.tv_sec;
	request->id_num = new_id();
	response->id_num = htonl(request->id_num);
	/* insert a new entry into the top of the list */
	ptr = (TABLE_ENTRY *)malloc(sizeof(TABLE_ENTRY));
	if (ptr == NIL) {
		syslog(LOG_ERR, "insert_table: Out of memory");
		_exit(1);
	}
	ptr->time = current_time;
	ptr->request = *request;
	ptr->next = table;
	if (ptr->next != NIL)
		ptr->next->last = ptr;
	ptr->last = NIL;
	table = ptr;
}

/*
 * Generate a unique non-zero sequence number
 */
int
new_id(void)
{
	static int current_id = 0;

	current_id = (current_id + 1) % MAX_ID;
	/* 0 is reserved, helps to pick up bugs */
	if (current_id == 0)
		current_id = 1;
	return (current_id);
}

/*
 * Delete the invitation with id 'id_num'
 */
int
delete_invite(u_int32_t id_num)
{
	TABLE_ENTRY *ptr;

	if (debug)
		syslog(LOG_DEBUG, "delete_invite(%d)", id_num);
	for (ptr = table; ptr != NIL; ptr = ptr->next) {
		if (ptr->request.id_num == id_num)
			break;
		if (debug)
			print_request("", &ptr->request);
	}
	if (ptr != NIL) {
		delete(ptr);
		return (SUCCESS);
	}
	return (NOT_HERE);
}

/*
 * Classic delete from a double-linked list
 */
static void
delete(TABLE_ENTRY *ptr)
{

	if (debug)
		print_request("delete", &ptr->request);
	if (table == ptr)
		table = ptr->next;
	else if (ptr->last != NIL)
		ptr->last->next = ptr->next;
	if (ptr->next != NIL)
		ptr->next->last = ptr->last;
	free((char *)ptr);
}
