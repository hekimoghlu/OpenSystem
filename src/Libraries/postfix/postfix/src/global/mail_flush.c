/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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

#include "sys_defs.h"

/* Utility library. */

/* Global library. */

#include <mail_params.h>
#include <mail_proto.h>
#include <mail_flush.h>

/* mail_flush_deferred - flush deferred/incoming queue */

int     mail_flush_deferred(void)
{
    static char qmgr_trigger[] = {
	QMGR_REQ_FLUSH_DEAD,		/* all hosts, all transports */
	QMGR_REQ_SCAN_ALL,		/* all time stamps */
	QMGR_REQ_SCAN_DEFERRED,		/* scan deferred queue */
	QMGR_REQ_SCAN_INCOMING,		/* scan incoming queue */
    };

    /*
     * Trigger the flush queue service.
     */
    return (mail_trigger(MAIL_CLASS_PUBLIC, var_queue_service,
			 qmgr_trigger, sizeof(qmgr_trigger)));
}

/* mail_flush_maildrop - flush maildrop queue */

int     mail_flush_maildrop(void)
{
    static char wakeup[] = {TRIGGER_REQ_WAKEUP};

    /*
     * Trigger the pickup service.
     */
    return (mail_trigger(MAIL_CLASS_PUBLIC, var_pickup_service,
			 wakeup, sizeof(wakeup)));
}
