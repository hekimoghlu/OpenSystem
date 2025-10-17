/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#include <name_mask.h>
#include <msg.h>

/* Global library. */

#include <mail_params.h>
#include <cleanup_user.h>
#include <mail_proto.h>
#include <int_filt.h>

/* int_filt_flags - map mail class to submission flags */

int     int_filt_flags(int class)
{
    static const NAME_MASK table[] = {
	MAIL_SRC_NAME_NOTIFY, MAIL_SRC_MASK_NOTIFY,
	MAIL_SRC_NAME_BOUNCE, MAIL_SRC_MASK_BOUNCE,
	MAIL_SRC_NAME_SENDMAIL, 0,
	MAIL_SRC_NAME_SMTPD, 0,
	MAIL_SRC_NAME_QMQPD, 0,
	MAIL_SRC_NAME_FORWARD, 0,
	MAIL_SRC_NAME_VERIFY, 0,
	0,
    };
    int     filtered_classes = 0;

    if (class && *var_int_filt_classes) {
	filtered_classes =
	    name_mask(VAR_INT_FILT_CLASSES, table, var_int_filt_classes);
	if (filtered_classes == 0)
	    msg_warn("%s: bad input: %s", VAR_INT_FILT_CLASSES,
		     var_int_filt_classes);
	if (filtered_classes & class)
	    return (CLEANUP_FLAG_FILTER | CLEANUP_FLAG_MILTER);
    }
    return (0);
}
