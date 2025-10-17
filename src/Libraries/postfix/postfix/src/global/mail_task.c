/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#include <safe.h>

/* Global library. */

#include "mail_params.h"
#include "mail_conf.h"
#include "mail_task.h"

/* mail_task - clean up and decorate the process name */

const char *mail_task(const char *argv0)
{
    static VSTRING *canon_name;
    const char *slash;
    const char *tag;

    if (canon_name == 0)
	canon_name = vstring_alloc(10);
    if ((slash = strrchr(argv0, '/')) != 0 && slash[1])
	argv0 = slash + 1;
    /* Setenv()-ed from main.cf, or inherited from master. */
    if ((tag = safe_getenv(CONF_ENV_LOGTAG)) == 0)
	/* Check main.cf settings directly, in case set-gid. */
	tag = var_syslog_name ? var_syslog_name :
	    mail_conf_eval(DEF_SYSLOG_NAME);
    vstring_sprintf(canon_name, "%s/%s", tag, argv0);
    return (vstring_str(canon_name));
}
