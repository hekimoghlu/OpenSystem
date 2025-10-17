/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
#include <stdbool.h>

#include <sys/systm.h>
#include <sys/sysctl.h>
#include <sys/sbuf.h>
#include <sys/types.h>
#include <sys/mcache.h>
#include <sys/malloc.h>

#include <os/log.h>

#include <net/nwk_wq.h>
#include <skywalk/lib/net_filter_event.h>

static uint32_t net_filter_event_state;
static bool net_filter_event_initialized;
static struct eventhandler_lists_ctxt net_filter_evhdlr_ctxt;

EVENTHANDLER_DECLARE(net_filter_event, net_filter_event_callback_t);

static struct sbuf *
net_filter_event_description(uint32_t state)
{
	struct sbuf *sbuf;

	sbuf = sbuf_new(NULL, NULL, 128, SBUF_AUTOEXTEND);
	if (state & NET_FILTER_EVENT_PF) {
		sbuf_cat(sbuf, "pf ");
	}
	if (state & NET_FILTER_EVENT_SOCKET) {
		sbuf_cat(sbuf, "socket ");
	}
	if (state & NET_FILTER_EVENT_INTERFACE) {
		sbuf_cat(sbuf, "interface ");
	}
	if (state & NET_FILTER_EVENT_IP) {
		sbuf_cat(sbuf, "ip ");
	}
	if (state & NET_FILTER_EVENT_ALF) {
		sbuf_cat(sbuf, "application-firewall ");
	}
	if (state & NET_FILTER_EVENT_PARENTAL_CONTROLS) {
		sbuf_cat(sbuf, "parental-controls ");
	}
	sbuf_trim(sbuf);
	sbuf_finish(sbuf);

	return sbuf;
}

static void
net_filter_event_callback(struct eventhandler_entry_arg arg0 __unused,
    enum net_filter_event_subsystems state)
{
	struct sbuf *sbuf = net_filter_event_description(state);

	os_log(OS_LOG_DEFAULT, "net_filter_event: new state (0x%x) %s",
	    state, sbuf_data(sbuf));
	evhlog(debug, "%s: eventhandler saw event type=net_filter_event_state event_code=%s",
	    __func__, sbuf_data(sbuf));
	sbuf_delete(sbuf);
}

static void
net_filter_event_init(void)
{
	if (net_filter_event_initialized) {
		return;
	}
	net_filter_event_initialized = true;
	eventhandler_lists_ctxt_init(&net_filter_evhdlr_ctxt);
	net_filter_event_register(net_filter_event_callback);
}

static void
net_filter_event_enqueue_callback(struct nwk_wq_entry *nwk_kwqe)
{
	EVENTHANDLER_INVOKE(&net_filter_evhdlr_ctxt, net_filter_event,
	    net_filter_event_state);
	kfree_type(struct nwk_wq_entry, nwk_kwqe);
}

static void
net_filter_event_enqueue(void)
{
	struct nwk_wq_entry *__single nwk_wqe;

	struct sbuf *sbuf = net_filter_event_description(net_filter_event_state);
	evhlog(debug, "%s: eventhandler enqueuing event of type=net_filter_event_state event_code=%s",
	    __func__, sbuf_data(sbuf));
	sbuf_delete(sbuf);

	nwk_wqe = kalloc_type(struct nwk_wq_entry, Z_WAITOK | Z_ZERO | Z_NOFAIL);
	nwk_wqe->func = net_filter_event_enqueue_callback;
	nwk_wq_enqueue(nwk_wqe);
}

void
net_filter_event_mark(enum net_filter_event_subsystems subsystem, bool compatible)
{
	uint32_t old_state = net_filter_event_state;

	net_filter_event_init();
	if (!compatible) {
		os_atomic_or(&net_filter_event_state, subsystem, relaxed);
	} else {
		os_atomic_andnot(&net_filter_event_state, subsystem, relaxed);
	}
	if (old_state != net_filter_event_state) {
		net_filter_event_enqueue();
	}
}

enum net_filter_event_subsystems
net_filter_event_get_state(void)
{
	return net_filter_event_state;
}

void
net_filter_event_register(net_filter_event_callback_t callback)
{
	net_filter_event_init();
	(void)EVENTHANDLER_REGISTER(&net_filter_evhdlr_ctxt,
	    net_filter_event, callback,
	    eventhandler_entry_dummy_arg,
	    EVENTHANDLER_PRI_ANY);
}

static int
net_filter_event_sysctl(struct sysctl_oid *oidp, void *arg1, int arg2,
    struct sysctl_req *req)
{
#pragma unused(oidp, arg1, arg2)
	struct sbuf *sbuf = net_filter_event_description(net_filter_event_state);

	int error = sysctl_io_string(req, sbuf_data(sbuf), 0, 0, NULL);
	sbuf_delete(sbuf);

	return error;
}

SYSCTL_PROC(_net, OID_AUTO, filter_state,
    CTLTYPE_STRING | CTLFLAG_LOCKED, NULL, 0,
    net_filter_event_sysctl, "A", "State of the network filters");
