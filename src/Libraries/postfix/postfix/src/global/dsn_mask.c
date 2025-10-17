/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

#include <name_code.h>
#include <name_mask.h>
#include <msg.h>

/* Global library. */

#include <dsn_mask.h>

/* Application-specific. */

static const NAME_MASK dsn_notify_table[] = {
    "NEVER", DSN_NOTIFY_NEVER,
    "SUCCESS", DSN_NOTIFY_SUCCESS,
    "FAILURE", DSN_NOTIFY_FAILURE,
    "DELAY", DSN_NOTIFY_DELAY,
    0, 0,
};

static const NAME_CODE dsn_ret_table[] = {
    "FULL", DSN_RET_FULL,
    "HDRS", DSN_RET_HDRS,
    0, 0,
};

/* dsn_ret_code - string to mask */

int     dsn_ret_code(const char *str)
{
    return (name_code(dsn_ret_table, NAME_CODE_FLAG_NONE, str));
}

/* dsn_ret_str - mask to string */

const char *dsn_ret_str(int code)
{
    const char *cp;

    if ((cp = str_name_code(dsn_ret_table, code)) == 0)
	msg_panic("dsn_ret_str: unknown code %d", code);
    return (cp);
}

/* dsn_notify_mask - string to mask */

int     dsn_notify_mask(const char *str)
{
    int     mask = name_mask_opt("DSN NOTIFY command", dsn_notify_table,
				 str, NAME_MASK_ANY_CASE | NAME_MASK_RETURN);

    return (DSN_NOTIFY_OK(mask) ? mask : 0);
}

/* dsn_notify_str - mask to string */

const char *dsn_notify_str(int mask)
{
    return (str_name_mask_opt((VSTRING *) 0, "DSN NOTIFY command",
			      dsn_notify_table, mask,
			      NAME_MASK_FATAL | NAME_MASK_COMMA));
}
