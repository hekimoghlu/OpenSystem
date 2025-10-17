/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

/* Utility library. */

#include <msg.h>

/* Global library. */

#include "mail_params.h"
#include "mail_conf.h"
#include "debug_process.h"

/* debug_process - run a debugger on this process */

void    debug_process(void)
{
    const char *command;

    /*
     * Expand $debugger_command then run it.
     */
    command = mail_conf_lookup_eval(VAR_DEBUG_COMMAND);
    if (command == 0 || *command == 0)
	msg_fatal("no %s variable set up", VAR_DEBUG_COMMAND);
    msg_info("running: %s", command);
    system(command);
}
