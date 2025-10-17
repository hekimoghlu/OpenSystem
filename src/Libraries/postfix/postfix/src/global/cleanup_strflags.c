/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include <vstring.h>

/* Global library. */

#include "cleanup_user.h"

 /*
  * Mapping from flags code to printable string.
  */
struct cleanup_flag_map {
    unsigned flag;
    const char *text;
};

static struct cleanup_flag_map cleanup_flag_map[] = {
    CLEANUP_FLAG_BOUNCE, "enable_bad_mail_bounce",
    CLEANUP_FLAG_FILTER, "enable_header_body_filter",
    CLEANUP_FLAG_HOLD, "hold_message",
    CLEANUP_FLAG_DISCARD, "discard_message",
    CLEANUP_FLAG_BCC_OK, "enable_automatic_bcc",
    CLEANUP_FLAG_MAP_OK, "enable_address_mapping",
    CLEANUP_FLAG_MILTER, "enable_milters",
    CLEANUP_FLAG_SMTP_REPLY, "enable_smtp_reply",
    CLEANUP_FLAG_SMTPUTF8, "smtputf8_requested",
    CLEANUP_FLAG_AUTOUTF8, "smtputf8_autodetect",
};

/* cleanup_strflags - map flags code to printable string */

const char *cleanup_strflags(unsigned flags)
{
    static VSTRING *result;
    unsigned i;

    if (flags == 0)
	return ("none");

    if (result == 0)
	result = vstring_alloc(20);
    else
	VSTRING_RESET(result);

    for (i = 0; i < sizeof(cleanup_flag_map) / sizeof(cleanup_flag_map[0]); i++) {
	if (cleanup_flag_map[i].flag & flags) {
	    vstring_sprintf_append(result, "%s ", cleanup_flag_map[i].text);
	    flags &= ~cleanup_flag_map[i].flag;
	}
    }

    if (flags != 0 || VSTRING_LEN(result) == 0)
	msg_panic("cleanup_strflags: unrecognized flag value(s) 0x%x", flags);

    vstring_truncate(result, VSTRING_LEN(result) - 1);
    VSTRING_TERMINATE(result);

    return (vstring_str(result));
}
