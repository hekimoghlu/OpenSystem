/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include <sys_defs.h>
#include <unistd.h>
#include <stdlib.h>

/* Utility library. */

#include <msg.h>
#include <iostuff.h>

/* Application-specific. */

#include <master.h>
#include <master_proto.h>

int     master_flow_pipe[2];

/* master_flow_init - initialize the flow control channel */

void    master_flow_init(void)
{
    const char *myname = "master_flow_init";

    if (pipe(master_flow_pipe) < 0)
	msg_fatal("%s: pipe: %m", myname);

    non_blocking(master_flow_pipe[0], NON_BLOCKING);
    non_blocking(master_flow_pipe[1], NON_BLOCKING);

    close_on_exec(master_flow_pipe[0], CLOSE_ON_EXEC);
    close_on_exec(master_flow_pipe[1], CLOSE_ON_EXEC);
}
