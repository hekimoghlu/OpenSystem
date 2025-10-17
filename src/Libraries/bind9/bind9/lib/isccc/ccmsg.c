/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
/* $Id: ccmsg.c,v 1.10 2007/08/28 07:20:43 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/mem.h>
#include <isc/result.h>
#include <isc/task.h>
#include <isc/util.h>

#include <isccc/events.h>
#include <isccc/ccmsg.h>

#define CCMSG_MAGIC		ISC_MAGIC('C', 'C', 'm', 's')
#define VALID_CCMSG(foo)	ISC_MAGIC_VALID(foo, CCMSG_MAGIC)

static void recv_length(isc_task_t *, isc_event_t *);
static void recv_message(isc_task_t *, isc_event_t *);


static void
recv_length(isc_task_t *task, isc_event_t *ev_in) {
	isc_socketevent_t *ev = (isc_socketevent_t *)ev_in;
	isc_event_t *dev;
	isccc_ccmsg_t *ccmsg = ev_in->ev_arg;
	isc_region_t region;
	isc_result_t result;

	INSIST(VALID_CCMSG(ccmsg));

	dev = &ccmsg->event;

	if (ev->result != ISC_R_SUCCESS) {
		ccmsg->result = ev->result;
		goto send_and_free;
	}

	/*
	 * Success.
	 */
	ccmsg->size = ntohl(ccmsg->size);
	if (ccmsg->size == 0) {
		ccmsg->result = ISC_R_UNEXPECTEDEND;
		goto send_and_free;
	}
	if (ccmsg->size > ccmsg->maxsize) {
		ccmsg->result = ISC_R_RANGE;
		goto send_and_free;
	}

	region.base = isc_mem_get(ccmsg->mctx, ccmsg->size);
	region.length = ccmsg->size;
	if (region.base == NULL) {
		ccmsg->result = ISC_R_NOMEMORY;
		goto send_and_free;
	}

	isc_buffer_init(&ccmsg->buffer, region.base, region.length);
	result = isc_socket_recv(ccmsg->sock, &region, 0,
				 task, recv_message, ccmsg);
	if (result != ISC_R_SUCCESS) {
		ccmsg->result = result;
		goto send_and_free;
	}

	isc_event_free(&ev_in);
	return;

 send_and_free:
	isc_task_send(ccmsg->task, &dev);
	ccmsg->task = NULL;
	isc_event_free(&ev_in);
	return;
}

static void
recv_message(isc_task_t *task, isc_event_t *ev_in) {
	isc_socketevent_t *ev = (isc_socketevent_t *)ev_in;
	isc_event_t *dev;
	isccc_ccmsg_t *ccmsg = ev_in->ev_arg;

	(void)task;

	INSIST(VALID_CCMSG(ccmsg));

	dev = &ccmsg->event;

	if (ev->result != ISC_R_SUCCESS) {
		ccmsg->result = ev->result;
		goto send_and_free;
	}

	ccmsg->result = ISC_R_SUCCESS;
	isc_buffer_add(&ccmsg->buffer, ev->n);
	ccmsg->address = ev->address;

 send_and_free:
	isc_task_send(ccmsg->task, &dev);
	ccmsg->task = NULL;
	isc_event_free(&ev_in);
}

void
isccc_ccmsg_init(isc_mem_t *mctx, isc_socket_t *sock, isccc_ccmsg_t *ccmsg) {
	REQUIRE(mctx != NULL);
	REQUIRE(sock != NULL);
	REQUIRE(ccmsg != NULL);

	ccmsg->magic = CCMSG_MAGIC;
	ccmsg->size = 0;
	ccmsg->buffer.base = NULL;
	ccmsg->buffer.length = 0;
	ccmsg->maxsize = 4294967295U;	/* Largest message possible. */
	ccmsg->mctx = mctx;
	ccmsg->sock = sock;
	ccmsg->task = NULL;			/* None yet. */
	ccmsg->result = ISC_R_UNEXPECTED;	/* None yet. */
	/*
	 * Should probably initialize the event here, but it can wait.
	 */
}


void
isccc_ccmsg_setmaxsize(isccc_ccmsg_t *ccmsg, unsigned int maxsize) {
	REQUIRE(VALID_CCMSG(ccmsg));

	ccmsg->maxsize = maxsize;
}


isc_result_t
isccc_ccmsg_readmessage(isccc_ccmsg_t *ccmsg,
		       isc_task_t *task, isc_taskaction_t action, void *arg)
{
	isc_result_t result;
	isc_region_t region;

	REQUIRE(VALID_CCMSG(ccmsg));
	REQUIRE(task != NULL);
	REQUIRE(ccmsg->task == NULL);  /* not currently in use */

	if (ccmsg->buffer.base != NULL) {
		isc_mem_put(ccmsg->mctx, ccmsg->buffer.base,
			    ccmsg->buffer.length);
		ccmsg->buffer.base = NULL;
		ccmsg->buffer.length = 0;
	}

	ccmsg->task = task;
	ccmsg->action = action;
	ccmsg->arg = arg;
	ccmsg->result = ISC_R_UNEXPECTED;  /* unknown right now */

	ISC_EVENT_INIT(&ccmsg->event, sizeof(isc_event_t), 0, 0,
		       ISCCC_EVENT_CCMSG, action, arg, ccmsg,
		       NULL, NULL);

	region.base = (unsigned char *)&ccmsg->size;
	region.length = 4;  /* isc_uint32_t */
	result = isc_socket_recv(ccmsg->sock, &region, 0,
				 ccmsg->task, recv_length, ccmsg);

	if (result != ISC_R_SUCCESS)
		ccmsg->task = NULL;

	return (result);
}

void
isccc_ccmsg_cancelread(isccc_ccmsg_t *ccmsg) {
	REQUIRE(VALID_CCMSG(ccmsg));

	isc_socket_cancel(ccmsg->sock, NULL, ISC_SOCKCANCEL_RECV);
}

#if 0
void
isccc_ccmsg_freebuffer(isccc_ccmsg_t *ccmsg) {
	REQUIRE(VALID_CCMSG(ccmsg));

	if (ccmsg->buffer.base == NULL)
		return;

	isc_mem_put(ccmsg->mctx, ccmsg->buffer.base, ccmsg->buffer.length);
	ccmsg->buffer.base = NULL;
	ccmsg->buffer.length = 0;
}
#endif

void
isccc_ccmsg_invalidate(isccc_ccmsg_t *ccmsg) {
	REQUIRE(VALID_CCMSG(ccmsg));

	ccmsg->magic = 0;

	if (ccmsg->buffer.base != NULL) {
		isc_mem_put(ccmsg->mctx, ccmsg->buffer.base,
			    ccmsg->buffer.length);
		ccmsg->buffer.base = NULL;
		ccmsg->buffer.length = 0;
	}
}
