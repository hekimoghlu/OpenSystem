/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
/*
  * System library.
  */
#include <sys_defs.h>
#include <netinet/in.h>			/* ntohs() for Solaris or BSD */
#include <arpa/inet.h>			/* ntohs() for Linux or BSD */
#include <string.h>

 /*
  * Utility library.
  */
#include <msg.h>
#include <vstring.h>
#include <base64_code.h>

 /*
  * Global library.
  */
#include <mail_params.h>

 /*
  * Application-specific.
  */
#include <smtp.h>

 /*
  * We use a configurable field terminator and optional place holder for data
  * that is unavailable or inapplicable. We base64-encode content that
  * contains these characters, and content that needs obfuscation.
  */

/* smtp_key_append_na - append place-holder key field */

static void smtp_key_append_na(VSTRING *buffer, const char *delim_na)
{
    if (delim_na[1] != 0)
	VSTRING_ADDCH(buffer, delim_na[1]);
    VSTRING_ADDCH(buffer, delim_na[0]);
}

/* smtp_key_append_str - append string-valued key field */

static void smtp_key_append_str(VSTRING *buffer, const char *str,
				        const char *delim_na)
{
    if (str == 0 || str[0] == 0) {
	smtp_key_append_na(buffer, delim_na);
    } else if (str[strcspn(str, delim_na)] != 0) {
	base64_encode_opt(buffer, str, strlen(str), BASE64_FLAG_APPEND);
	VSTRING_ADDCH(buffer, delim_na[0]);
    } else {
	vstring_sprintf_append(buffer, "%s%c", str, delim_na[0]);
    }
}

/* smtp_key_append_uint - append unsigned-valued key field */

static void smtp_key_append_uint(VSTRING *buffer, unsigned num,
				         const char *delim_na)
{
    vstring_sprintf_append(buffer, "%u%c", num, delim_na[0]);
}

/* smtp_key_prefix - format common elements in lookup key */

char   *smtp_key_prefix(VSTRING *buffer, const char *delim_na,
			        SMTP_ITERATOR *iter, int flags)
{
    static const char myname[] = "smtp_key_prefix";
    SMTP_STATE *state = iter->parent;	/* private member */

    /*
     * Sanity checks.
     */
    if (state == 0)
	msg_panic("%s: no parent state", myname);
    if (flags & ~SMTP_KEY_MASK_ALL)
	msg_panic("%s: unknown key flags 0x%x",
		  myname, flags & ~SMTP_KEY_MASK_ALL);
    if (flags == 0)
	msg_panic("%s: zero flags", myname);

    /*
     * Initialize.
     */
    VSTRING_RESET(buffer);

    /*
     * Per-service and per-request context.
     */
    if (flags & SMTP_KEY_FLAG_SERVICE)
	smtp_key_append_str(buffer, state->service, delim_na);
    if (flags & SMTP_KEY_FLAG_SENDER)
	smtp_key_append_str(buffer, state->request->sender, delim_na);

    /*
     * Per-destination context, non-canonicalized form.
     */
    if (flags & SMTP_KEY_FLAG_REQ_NEXTHOP)
	smtp_key_append_str(buffer, STR(iter->request_nexthop), delim_na);
    if (flags & SMTP_KEY_FLAG_NEXTHOP)
	smtp_key_append_str(buffer, STR(iter->dest), delim_na);

    /*
     * Per-host context, canonicalized form.
     */
    if (flags & SMTP_KEY_FLAG_HOSTNAME)
	smtp_key_append_str(buffer, STR(iter->host), delim_na);
    if (flags & SMTP_KEY_FLAG_ADDR)
	smtp_key_append_str(buffer, STR(iter->addr), delim_na);
    if (flags & SMTP_KEY_FLAG_PORT)
	smtp_key_append_uint(buffer, ntohs(iter->port), delim_na);

    /* Similarly, provide unique TLS fingerprint when applicable. */

    VSTRING_TERMINATE(buffer);

    return STR(buffer);
}
