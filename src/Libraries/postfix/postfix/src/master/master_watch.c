/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
#include <unistd.h>

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>

/* Application-specific. */

#include "master.h"

/* master_str_watch - watch string-valued parameters for change */

void    master_str_watch(const MASTER_STR_WATCH *str_watch_table)
{
    const MASTER_STR_WATCH *wp;

    for (wp = str_watch_table; wp->name != 0; wp++) {

	/*
	 * Detect changes to monitored parameter values. If a change is
	 * supported, we discard the backed up value and update it to the
	 * current value later. Otherwise we complain.
	 */
	if (wp->backup[0] != 0
	    && strcmp(wp->backup[0], wp->value[0]) != 0) {
	    if ((wp->flags & MASTER_WATCH_FLAG_UPDATABLE) == 0) {
		msg_warn("ignoring %s parameter value change", wp->name);
		msg_warn("old value: \"%s\", new value: \"%s\"",
			 wp->backup[0], wp->value[0]);
		msg_warn("to change %s, stop and start Postfix", wp->name);
	    } else {
		myfree(wp->backup[0]);
		wp->backup[0] = 0;
	    }
	}

	/*
	 * Initialize the backed up parameter value, or update it if this
	 * parameter supports updates after initialization. Optionally 
	 * notify the application that this parameter has changed.
	 */
	if (wp->backup[0] == 0) {
	    if (wp->notify != 0)
		wp->notify();
	    wp->backup[0] = mystrdup(wp->value[0]);
	}
    }
}

/* master_int_watch - watch integer-valued parameters for change */

void    master_int_watch(MASTER_INT_WATCH *int_watch_table)
{
    MASTER_INT_WATCH *wp;

    for (wp = int_watch_table; wp->name != 0; wp++) {

	/*
	 * Detect changes to monitored parameter values. If a change is
	 * supported, we discard the backed up value and update it to the
	 * current value later. Otherwise we complain.
	 */
	if ((wp->flags & MASTER_WATCH_FLAG_ISSET) != 0
	    && wp->backup != wp->value[0]) {
	    if ((wp->flags & MASTER_WATCH_FLAG_UPDATABLE) == 0) {
		msg_warn("ignoring %s parameter value change", wp->name);
		msg_warn("old value: \"%d\", new value: \"%d\"",
			 wp->backup, wp->value[0]);
		msg_warn("to change %s, stop and start Postfix", wp->name);
	    } else {
		wp->flags &= ~MASTER_WATCH_FLAG_ISSET;
	    }
	}

	/*
	 * Initialize the backed up parameter value, or update if it this
	 * parameter supports updates after initialization. Optionally 
	 * notify the application that this parameter has changed.
	 */
	if ((wp->flags & MASTER_WATCH_FLAG_ISSET) == 0) {
	    if (wp->notify != 0)
		wp->notify();
	    wp->flags |= MASTER_WATCH_FLAG_ISSET;
	    wp->backup = wp->value[0];
	}
    }
}
