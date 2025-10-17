/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
#include <smtputf8.h>

/* smtputf8_autodetect - enable SMTPUTF8 autodetection */

int     smtputf8_autodetect(int class)
{
    static const char myname[] = "smtputf8_autodetect";
    static const NAME_MASK table[] = {
	MAIL_SRC_NAME_SENDMAIL, MAIL_SRC_MASK_SENDMAIL,
	MAIL_SRC_NAME_SMTPD, MAIL_SRC_MASK_SMTPD,
	MAIL_SRC_NAME_QMQPD, MAIL_SRC_MASK_QMQPD,
	MAIL_SRC_NAME_FORWARD, MAIL_SRC_MASK_FORWARD,
	MAIL_SRC_NAME_BOUNCE, MAIL_SRC_MASK_BOUNCE,
	MAIL_SRC_NAME_NOTIFY, MAIL_SRC_MASK_NOTIFY,
	MAIL_SRC_NAME_VERIFY, MAIL_SRC_MASK_VERIFY,
	MAIL_SRC_NAME_ALL, MAIL_SRC_MASK_ALL,
	0,
    };
    int     autodetect_classes = 0;

    if (class == 0 || (class & ~MAIL_SRC_MASK_ALL) != 0)
	msg_panic("%s: bad source class: %d", myname, class);
    if (*var_smtputf8_autoclass) {
	autodetect_classes =
	    name_mask(VAR_SMTPUTF8_AUTOCLASS, table, var_smtputf8_autoclass);
	if (autodetect_classes == 0)
	    msg_warn("%s: bad input: %s", VAR_SMTPUTF8_AUTOCLASS,
		     var_smtputf8_autoclass);
	if (autodetect_classes & class)
	    return (CLEANUP_FLAG_AUTOUTF8);
    }
    return (0);
}
