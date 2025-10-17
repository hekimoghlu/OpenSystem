/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#include <signal.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstring.h>

/* Global library. */

#include <mail_queue.h>

/* Application-specific. */

#include "bounce_service.h"

 /*
  * Support for removing a logfile when an update fails. In order to do this,
  * we save a copy of the currently-open logfile name, and register a
  * callback function pointer with the run-time error handler. The saved
  * pathname is made global so that the application can see whether or not a
  * trap was set up.
  */
static MSG_CLEANUP_FN bounce_cleanup_func;	/* saved callback */
VSTRING *bounce_cleanup_path;		/* saved path name */

/* bounce_cleanup_callback - run-time callback to cleanup logfile */

static void bounce_cleanup_callback(void)
{

    /*
     * Remove the logfile.
     */
    if (bounce_cleanup_path)
	bounce_cleanup_log();

    /*
     * Execute the saved cleanup action.
     */
    if (bounce_cleanup_func)
	bounce_cleanup_func();
}

/* bounce_cleanup_log - clean up the logfile */

void    bounce_cleanup_log(void)
{
    const char *myname = "bounce_cleanup_log";

    /*
     * Sanity checks.
     */
    if (bounce_cleanup_path == 0)
	msg_panic("%s: no cleanup context", myname);

    /*
     * This function may be called before a logfile is created or after it
     * has been deleted, so do not complain.
     */
    (void) unlink(vstring_str(bounce_cleanup_path));
}

/* bounce_cleanup_sig - signal handler */

static void bounce_cleanup_sig(int sig)
{

    /*
     * Running as a signal handler - don't do complicated stuff.
     */
    if (bounce_cleanup_path)
	(void) unlink(vstring_str(bounce_cleanup_path));
    _exit(sig);
}

/* bounce_cleanup_register - register logfile to clean up */

void    bounce_cleanup_register(char *service, char *queue_id)
{
    const char *myname = "bounce_cleanup_register";

    /*
     * Sanity checks.
     */
    if (bounce_cleanup_path)
	msg_panic("%s: nested call", myname);

    /*
     * Save a copy of the logfile path, and of the last callback function
     * pointer registered with the run-time error handler.
     */
    bounce_cleanup_path = vstring_alloc(10);
    (void) mail_queue_path(bounce_cleanup_path, service, queue_id);
    bounce_cleanup_func = msg_cleanup(bounce_cleanup_callback);
    signal(SIGTERM, bounce_cleanup_sig);
}

/* bounce_cleanup_unregister - unregister logfile to clean up */

void    bounce_cleanup_unregister(void)
{
    const char *myname = "bounce_cleanup_unregister";

    /*
     * Sanity checks.
     */
    if (bounce_cleanup_path == 0)
	msg_panic("%s: no cleanup context", myname);

    /*
     * Restore the saved callback function pointer, and release storage for
     * the saved logfile pathname.
     */
    signal(SIGTERM, SIG_DFL);
    (void) msg_cleanup(bounce_cleanup_func);
    vstring_free(bounce_cleanup_path);
    bounce_cleanup_path = 0;
}
