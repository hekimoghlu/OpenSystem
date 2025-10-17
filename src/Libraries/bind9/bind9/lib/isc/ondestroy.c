/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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
/* $Id: ondestroy.c,v 1.16 2007/06/19 23:47:17 tbox Exp $ */

/*! \file */

#include <config.h>

#include <stddef.h>

#include <isc/event.h>
#include <isc/magic.h>
#include <isc/ondestroy.h>
#include <isc/task.h>
#include <isc/util.h>

#define ONDESTROY_MAGIC		ISC_MAGIC('D', 'e', 'S', 't')
#define VALID_ONDESTROY(s)	ISC_MAGIC_VALID(s, ONDESTROY_MAGIC)

void
isc_ondestroy_init(isc_ondestroy_t *ondest) {
	ondest->magic = ONDESTROY_MAGIC;
	ISC_LIST_INIT(ondest->events);
}

isc_result_t
isc_ondestroy_register(isc_ondestroy_t *ondest, isc_task_t *task,
		       isc_event_t **eventp)
{
	isc_event_t *theevent;
	isc_task_t *thetask = NULL;

	REQUIRE(VALID_ONDESTROY(ondest));
	REQUIRE(task != NULL);
	REQUIRE(eventp != NULL);

	theevent = *eventp;

	REQUIRE(theevent != NULL);

	isc_task_attach(task, &thetask);

	theevent->ev_sender = thetask;

	ISC_LIST_APPEND(ondest->events, theevent, ev_link);

	return (ISC_R_SUCCESS);
}

void
isc_ondestroy_notify(isc_ondestroy_t *ondest, void *sender) {
	isc_event_t *eventp;
	isc_task_t *task;

	REQUIRE(VALID_ONDESTROY(ondest));

	eventp = ISC_LIST_HEAD(ondest->events);
	while (eventp != NULL) {
		ISC_LIST_UNLINK(ondest->events, eventp, ev_link);

		task = eventp->ev_sender;
		eventp->ev_sender = sender;

		isc_task_sendanddetach(&task, &eventp);

		eventp = ISC_LIST_HEAD(ondest->events);
	}
}


