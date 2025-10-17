/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <string.h>
#include <syslog.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"

/*
 * For converting between syslog numbers and strings.
 */
struct strmap {
    const char *name;
    int num;
};

static const struct strmap priorities[] = {
	{ "alert",	LOG_ALERT },
	{ "crit",	LOG_CRIT },
	{ "debug",	LOG_DEBUG },
	{ "emerg",	LOG_EMERG },
	{ "err",	LOG_ERR },
	{ "info",	LOG_INFO },
	{ "notice",	LOG_NOTICE },
	{ "warning",	LOG_WARNING },
	{ "none",	-1 },
	{ NULL,		-1 }
};

bool
sudo_str2logpri_v1(const char *str, int *logpri)
{
    const struct strmap *pri;
    debug_decl(sudo_str2logpri, SUDO_DEBUG_UTIL);

    for (pri = priorities; pri->name != NULL; pri++) {
	if (strcmp(str, pri->name) == 0) {
	    *logpri = pri->num;
	    debug_return_bool(true);
	}
    }
    debug_return_bool(false);
}

const char *
sudo_logpri2str_v1(int num)
{
    const struct strmap *pri;
    debug_decl(sudo_logpri2str, SUDO_DEBUG_UTIL);

    for (pri = priorities; pri->name != NULL; pri++) {
	if (pri->num == num)
	    break;
    }
    debug_return_const_str(pri->name);
}
