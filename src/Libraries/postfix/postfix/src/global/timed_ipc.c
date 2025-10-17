/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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

/* Utility library. */

#include <msg.h>
#include <vstream.h>

/* Global library. */

#include "mail_params.h"
#include "timed_ipc.h"

/* timed_ipc_setup - enable ipc with timeout */

void    timed_ipc_setup(VSTREAM *stream)
{
    if (var_ipc_timeout <= 0)
	msg_panic("timed_ipc_setup: bad ipc_timeout %d", var_ipc_timeout);

    vstream_control(stream,
		    CA_VSTREAM_CTL_TIMEOUT(var_ipc_timeout),
		    CA_VSTREAM_CTL_END);
}
