/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include <ctype.h>
#include <string.h>

/* Utility library. */

#include <msg.h>

/* Global library. */

/* Application-specific. */

#include <smtpd_dsn_fix.h>

struct dsn_map {
    const char *micro_code;		/* Final digits in mailbox D.S.N. */
    const char *sender_dsn;		/* Replacement sender D.S.N. */
    const char *rcpt_dsn;		/* Replacement recipient D.S.N. */
};

static struct dsn_map dsn_map[] = {
    /* - Sender - Recipient */
    "1", SND_DSN, "4.1.1",		/* 4.1.1: Bad dest mbox addr */
    "2", "4.1.8", "4.1.2",		/* 4.1.2: Bad dest system addr */
    "3", "4.1.7", "4.1.3",		/* 4.1.3: Bad dest mbox addr syntax */
    "4", SND_DSN, "4.1.4",		/* 4.1.4: Dest mbox addr ambiguous */
    "5", "4.1.0", "4.1.5",		/* 4.1.5: Dest mbox addr valid */
    "6", SND_DSN, "4.1.6",		/* 4.1.6: Mailbox has moved */
    "7", "4.1.7", "4.1.3",		/* 4.1.7: Bad sender mbox addr syntax */
    "8", "4.1.8", "4.1.2",		/* 4.1.8: Bad sender system addr */
    0, "4.1.0", "4.1.0",		/* Default mapping */
};

/* smtpd_dsn_fix - fix DSN status */

const char *smtpd_dsn_fix(const char *status, const char *reply_class)
{
    struct dsn_map *dp;
    const char *result = status;

    /*
     * Update an address-specific DSN according to what is being rejected.
     */
    if (ISDIGIT(status[0]) && strncmp(status + 1, ".1.", 3) == 0) {

	/*
	 * Fix recipient address DSN while rejecting a sender address. Don't
	 * let future recipient-specific DSN codes slip past us.
	 */
	if (strcmp(reply_class, SMTPD_NAME_SENDER) == 0) {
	    for (dp = dsn_map; dp->micro_code != 0; dp++)
		if (strcmp(status + 4, dp->micro_code) == 0)
		    break;
	    result = dp->sender_dsn;
	}

	/*
	 * Fix sender address DSN while rejecting a recipient address. Don't
	 * let future sender-specific DSN codes slip past us.
	 */
	else if (strcmp(reply_class, SMTPD_NAME_RECIPIENT) == 0) {
	    for (dp = dsn_map; dp->micro_code != 0; dp++)
		if (strcmp(status + 4, dp->micro_code) == 0)
		    break;
	    result = dp->rcpt_dsn;
	}

	/*
	 * Fix address-specific DSN while rejecting a non-address.
	 */
	else {
	    result = "4.0.0";
	}

	/*
	 * Give them a clue of what is going on.
	 */
	if (strcmp(status + 2, result + 2) != 0)
	    msg_info("mapping DSN status %s into %s status %c%s",
		     status, reply_class, status[0], result + 1);
	return (result);
    }

    /*
     * Don't update a non-address DSN. There are many legitimate uses for
     * these while rejecting address or non-address information.
     */
    else {
	return (status);
    }
}
