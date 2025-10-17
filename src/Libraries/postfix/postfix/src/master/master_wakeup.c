/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
/* System library. */

#include <sys_defs.h>
#include <unistd.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <trigger.h>
#include <events.h>
#include <set_eugid.h>
#include <set_ugid.h>

/* Global library. */

#include <mail_proto.h>			/* triggers */
#include <mail_params.h>

/* Application-specific. */

#include "mail_server.h"
#include "master.h"

/* master_wakeup_timer_event - wakeup event handler */

static void master_wakeup_timer_event(int unused_event, void *context)
{
    const char *myname = "master_wakeup_timer_event";
    MASTER_SERV *serv = (MASTER_SERV *) context;
    int     status;
    static char wakeup = TRIGGER_REQ_WAKEUP;

    /*
     * Don't wakeup services whose automatic wakeup feature was turned off in
     * the mean time.
     */
    if (serv->wakeup_time == 0)
	return;

    /*
     * Don't wake up services that are throttled. Find out what transport to
     * use. We can't block here so we choose a short timeout.
     */
#define BRIEFLY	1

    if (MASTER_THROTTLED(serv) == 0) {
	if (msg_verbose)
	    msg_info("%s: service %s", myname, serv->name);

	switch (serv->type) {
	case MASTER_SERV_TYPE_INET:
	    status = inet_trigger(serv->name, &wakeup, sizeof(wakeup), BRIEFLY);
	    break;
	case MASTER_SERV_TYPE_UNIX:
	    status = LOCAL_TRIGGER(serv->name, &wakeup, sizeof(wakeup), BRIEFLY);
	    break;
#ifdef MASTER_SERV_TYPE_PASS
	case MASTER_SERV_TYPE_PASS:
	    status = pass_trigger(serv->name, &wakeup, sizeof(wakeup), BRIEFLY);
	    break;
#endif

	    /*
	     * If someone compromises the postfix account then this must not
	     * overwrite files outside the chroot jail. Countermeasures:
	     * 
	     * - Limit the damage by accessing the FIFO as postfix not root.
	     * 
	     * - Have fifo_trigger() call safe_open() so we won't follow
	     * arbitrary hard/symlinks to files in/outside the chroot jail.
	     * 
	     * - All non-chroot postfix-related files must be root owned (or
	     * postfix check complains).
	     * 
	     * - The postfix user and group ID must not be shared with other
	     * applications (says the INSTALL documentation).
	     * 
	     * Result of a discussion with Michael Tokarev, who received his
	     * insights from Solar Designer, who tested Postfix with a kernel
	     * module that is paranoid about open() calls.
	     */
	case MASTER_SERV_TYPE_FIFO:
	    set_eugid(var_owner_uid, var_owner_gid);
	    status = fifo_trigger(serv->name, &wakeup, sizeof(wakeup), BRIEFLY);
	    set_ugid(getuid(), getgid());
	    break;
	default:
	    msg_panic("%s: unknown service type: %d", myname, serv->type);
	}
	if (status < 0)
	    msg_warn("%s: service %s(%s): %m",
		     myname, serv->ext_name, serv->name);
    }

    /*
     * Schedule another wakeup event.
     */
    event_request_timer(master_wakeup_timer_event, (void *) serv,
			serv->wakeup_time);
}

/* master_wakeup_init - start automatic service wakeup */

void    master_wakeup_init(MASTER_SERV *serv)
{
    const char *myname = "master_wakeup_init";

    if (serv->wakeup_time == 0 || (serv->flags & MASTER_FLAG_CONDWAKE))
	return;
    if (msg_verbose)
	msg_info("%s: service %s time %d",
		 myname, serv->name, serv->wakeup_time);
    master_wakeup_timer_event(0, (void *) serv);
}

/* master_wakeup_cleanup - cancel wakeup timer */

void    master_wakeup_cleanup(MASTER_SERV *serv)
{
    const char *myname = "master_wakeup_cleanup";

    /*
     * Cleanup, even when the wakeup feature has been turned off. There might
     * still be a pending timer. Don't depend on the code that reloads the
     * config file to reset the wakeup timer when things change.
     */
    if (msg_verbose)
	msg_info("%s: service %s", myname, serv->name);

    event_cancel_timer(master_wakeup_timer_event, (void *) serv);
}
