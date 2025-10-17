/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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

#include <stringops.h>

/* Global library. */

#include "mail_params.h"
#include "mail_addr.h"

/* mail_addr_double_bounce - construct the local double-bounce address */

const char *mail_addr_double_bounce(void)
{
    static char *addr;

    if (addr == 0)
	addr = concatenate(var_double_bounce_sender, "@",
			   var_myhostname, (char *) 0);
    return (addr);
}

/* mail_addr_postmaster - construct the local postmaster address */

const char *mail_addr_postmaster(void)
{
    static char *addr;

    if (addr == 0)
	addr = concatenate(MAIL_ADDR_POSTMASTER, "@",
			   var_myhostname, (char *) 0);
    return (addr);
}

/* mail_addr_mail_daemon - construct the local mailer-daemon address */

const char *mail_addr_mail_daemon(void)
{
    static char *addr;

    if (addr == 0)
	addr = concatenate(MAIL_ADDR_MAIL_DAEMON, "@",
			   var_myhostname, (char *) 0);
    return (addr);
}
