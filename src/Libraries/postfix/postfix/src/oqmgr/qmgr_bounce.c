/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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

/* Global library. */

#include <bounce.h>
#include <deliver_completed.h>

/* Application-specific. */

#include "qmgr.h"

/* qmgr_bounce_recipient - bounce one message recipient */

void    qmgr_bounce_recipient(QMGR_MESSAGE *message, RECIPIENT *recipient,
			              DSN *dsn)
{
    MSG_STATS stats;
    int     status;

    status = bounce_append(message->tflags, message->queue_id,
			   QMGR_MSG_STATS(&stats, message), recipient,
			   "none", dsn);

    if (status == 0)
	deliver_completed(message->fp, recipient->offset);
    else
	message->flags |= status;
}
