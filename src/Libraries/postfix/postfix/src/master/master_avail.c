/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
/* System libraries. */

#include <sys_defs.h>

/* Utility library. */

#include <events.h>
#include <msg.h>

/* Application-specific. */

#include "master_proto.h"
#include "master.h"

/* master_avail_event - create child process to handle connection request */

static void master_avail_event(int event, void *context)
{
    MASTER_SERV *serv = (MASTER_SERV *) context;
    time_t  now;

    if (event == 0)				/* XXX Can this happen? */
	msg_panic("master_avail_event: null event");
    else {

	/*
	 * When all servers for a public internet service are busy, we start
	 * creating server processes with "-o stress=yes" on the command
	 * line, and keep creating such processes until the process count is
	 * below the limit for at least 1000 seconds. This provides a minimal
	 * solution that can be adopted into legacy and stable Postfix
	 * releases.
	 * 
	 * This is not the right place to update serv->stress_param_val in
	 * response to stress level changes. Doing so would would contaminate
	 * the "postfix reload" code with stress management implementation
	 * details, creating a source of future bugs. Instead, we update
	 * simple counters or flags here, and use their values to determine
	 * the proper serv->stress_param_val value when exec-ing a server
	 * process.
	 */
	if (serv->stress_param_val != 0
	    && !MASTER_LIMIT_OK(serv->max_proc, serv->total_proc + 1)) {
	    now = event_time();
	    if (serv->stress_expire_time < now)
		master_restart_service(serv, NO_CONF_RELOAD);
	    serv->stress_expire_time = now + 1000;
	}
	master_spawn(serv);
    }
}

/* master_avail_listen - enforce the socket monitoring policy */

void    master_avail_listen(MASTER_SERV *serv)
{
    const char *myname = "master_avail_listen";
    int     listen_flag;
    time_t  now;
    int     n;

    /*
     * Caution: several other master_XXX modules call master_avail_listen(),
     * master_avail_more() or master_avail_less(). To avoid mutual dependency
     * problems, the code below invokes no code in other master_XXX modules,
     * and modifies no data that is maintained by other master_XXX modules.
     * 
     * When no-one else is monitoring the service's listen socket, start
     * monitoring the socket for connection requests. All this under the
     * restriction that we have sufficient resources to service a connection
     * request.
     */
    if (msg_verbose)
	msg_info("%s: %s avail %d total %d max %d", myname, serv->name,
		 serv->avail_proc, serv->total_proc, serv->max_proc);
    if (MASTER_THROTTLED(serv) || serv->avail_proc > 0) {
	listen_flag = 0;
    } else if (MASTER_LIMIT_OK(serv->max_proc, serv->total_proc)) {
	listen_flag = 1;
    } else {
	listen_flag = 0;
	if (serv->stress_param_val != 0) {
	    now = event_time();
	    if (serv->busy_warn_time < now - 1000) {
		serv->busy_warn_time = now;
		msg_warn("service \"%s\" (%s) has reached its process limit \"%d\": "
			 "new clients may experience noticeable delays",
			 serv->ext_name, serv->name, serv->max_proc);
		msg_warn("to avoid this condition, increase the process count "
		      "in master.cf or reduce the service time per client");
		msg_warn("see http://www.postfix.org/STRESS_README.html for "
		      "examples of stress-adapting configuration settings");
	    }
	}
    }
    if (listen_flag && !MASTER_LISTENING(serv)) {
	if (msg_verbose)
	    msg_info("%s: enable events %s", myname, serv->name);
	for (n = 0; n < serv->listen_fd_count; n++)
	    event_enable_read(serv->listen_fd[n], master_avail_event,
			      (void *) serv);
	serv->flags |= MASTER_FLAG_LISTEN;
    } else if (!listen_flag && MASTER_LISTENING(serv)) {
	if (msg_verbose)
	    msg_info("%s: disable events %s", myname, serv->name);
	for (n = 0; n < serv->listen_fd_count; n++)
	    event_disable_readwrite(serv->listen_fd[n]);
	serv->flags &= ~MASTER_FLAG_LISTEN;
    }
}

/* master_avail_cleanup - cleanup */

void    master_avail_cleanup(MASTER_SERV *serv)
{
    int     n;

    master_delete_children(serv);		/* XXX calls
						 * master_avail_listen */

    /*
     * This code is redundant because master_delete_children() throttles the
     * service temporarily before calling master_avail_listen/less(), which
     * then turn off read events. This temporary throttling is not documented
     * (it is only an optimization), and therefore we must not depend on it.
     */
    if (MASTER_LISTENING(serv)) {
	for (n = 0; n < serv->listen_fd_count; n++)
	    event_disable_readwrite(serv->listen_fd[n]);
	serv->flags &= ~MASTER_FLAG_LISTEN;
    }
}

/* master_avail_more - one more available child process */

void    master_avail_more(MASTER_SERV *serv, MASTER_PROC *proc)
{
    const char *myname = "master_avail_more";

    /*
     * Caution: several other master_XXX modules call master_avail_listen(),
     * master_avail_more() or master_avail_less(). To avoid mutual dependency
     * problems, the code below invokes no code in other master_XXX modules,
     * and modifies no data that is maintained by other master_XXX modules.
     * 
     * This child process has become available for servicing connection
     * requests, so we can stop monitoring the service's listen socket. The
     * child will do it for us.
     */
    if (msg_verbose)
	msg_info("%s: pid %d (%s)", myname, proc->pid, proc->serv->name);
    if (proc->avail == MASTER_STAT_AVAIL)
	msg_panic("%s: process already available", myname);
    serv->avail_proc++;
    proc->avail = MASTER_STAT_AVAIL;
    master_avail_listen(serv);
}

/* master_avail_less - one less available child process */

void    master_avail_less(MASTER_SERV *serv, MASTER_PROC *proc)
{
    const char *myname = "master_avail_less";

    /*
     * Caution: several other master_XXX modules call master_avail_listen(),
     * master_avail_more() or master_avail_less(). To avoid mutual dependency
     * problems, the code below invokes no code in other master_XXX modules,
     * and modifies no data that is maintained by other master_XXX modules.
     * 
     * This child is no longer available for servicing connection requests. When
     * no child processes are available, start monitoring the service's
     * listen socket for new connection requests.
     */
    if (msg_verbose)
	msg_info("%s: pid %d (%s)", myname, proc->pid, proc->serv->name);
    if (proc->avail != MASTER_STAT_AVAIL)
	msg_panic("%s: process not available", myname);
    serv->avail_proc--;
    proc->avail = MASTER_STAT_TAKEN;
    master_avail_listen(serv);
}
