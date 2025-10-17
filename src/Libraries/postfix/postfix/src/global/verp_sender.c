/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

/* Utility library. */

#include <vstring.h>

/* Global library. */

#include <mail_params.h>
#include <recipient_list.h>
#include <verp_sender.h>

/* verp_sender - encode recipient into envelope sender address */

VSTRING *verp_sender(VSTRING *buf, const char *delimiters,
		             const char *sender, const RECIPIENT *rcpt_info)
{
    ssize_t send_local_len;
    ssize_t rcpt_local_len;
    const char *recipient;
    const char *cp;

    /*
     * Change prefix@origin into prefix+user=domain@origin.
     * 
     * Fix 20090115: Use the Postfix original recipient, because that is what
     * the VERP consumer expects.
     */
    send_local_len = ((cp = strrchr(sender, '@')) != 0 ?
		      cp - sender : strlen(sender));
    recipient = (rcpt_info->orig_addr[0] ?
		 rcpt_info->orig_addr : rcpt_info->address);
    rcpt_local_len = ((cp = strrchr(recipient, '@')) != 0 ?
		      cp - recipient : strlen(recipient));
    vstring_strncpy(buf, sender, send_local_len);
    VSTRING_ADDCH(buf, delimiters[0] & 0xff);
    vstring_strncat(buf, recipient, rcpt_local_len);
    if (recipient[rcpt_local_len] && recipient[rcpt_local_len + 1]) {
	VSTRING_ADDCH(buf, delimiters[1] & 0xff);
	vstring_strcat(buf, recipient + rcpt_local_len + 1);
    }
    if (sender[send_local_len] && sender[send_local_len + 1]) {
	VSTRING_ADDCH(buf, '@');
	vstring_strcat(buf, sender + send_local_len + 1);
    }
    VSTRING_TERMINATE(buf);
    return (buf);
}

/* verp_delims_verify - sanitize VERP delimiters */

const char *verp_delims_verify(const char *delims)
{
    if (strlen(delims) != 2)
	return ("bad VERP delimiter character count");
    if (strchr(var_verp_filter, delims[0]) == 0)
	return ("bad first VERP delimiter character");
    if (strchr(var_verp_filter, delims[1]) == 0)
	return ("bad second VERP delimiter character");
    return (0);
}
