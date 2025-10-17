/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#include <string.h>
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>

/* Application-specific. */

#include "master.h"

MASTER_SERV *master_head;

/* master_start_service - activate service */

void    master_start_service(MASTER_SERV *serv)
{

    /*
     * Enable connection requests, wakeup timers, and status updates from
     * child processes.
     */
    master_listen_init(serv);
    master_avail_listen(serv);
    master_status_init(serv);
    master_wakeup_init(serv);
}

/* master_stop_service - deactivate service */

void    master_stop_service(MASTER_SERV *serv)
{

    /*
     * Undo the things that master_start_service() did.
     */
    master_wakeup_cleanup(serv);
    master_status_cleanup(serv);
    master_avail_cleanup(serv);
    master_listen_cleanup(serv);
}

/* master_restart_service - restart service after configuration reload */

void    master_restart_service(MASTER_SERV *serv, int conf_reload)
{

    /*
     * Undo some of the things that master_start_service() did.
     */
    master_wakeup_cleanup(serv);
    master_status_cleanup(serv);

    /*
     * Now undo the undone.
     */
    master_status_init(serv);
    master_wakeup_init(serv);

    /*
     * Respond to configuration change.
     */
    if (conf_reload)
	master_avail_listen(serv);
}
