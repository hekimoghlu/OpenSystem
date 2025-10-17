/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

static const struct strmap facilities[] = {
#ifdef LOG_AUTHPRIV
	{ "authpriv",	LOG_AUTHPRIV },
#endif
	{ "auth",	LOG_AUTH },
	{ "daemon",	LOG_DAEMON },
	{ "user",	LOG_USER },
	{ "local0",	LOG_LOCAL0 },
	{ "local1",	LOG_LOCAL1 },
	{ "local2",	LOG_LOCAL2 },
	{ "local3",	LOG_LOCAL3 },
	{ "local4",	LOG_LOCAL4 },
	{ "local5",	LOG_LOCAL5 },
	{ "local6",	LOG_LOCAL6 },
	{ "local7",	LOG_LOCAL7 },
	{ NULL,		-1 }
};

bool
sudo_str2logfac_v1(const char *str, int *logfac)
{
    const struct strmap *fac;
    debug_decl(sudo_str2logfac, SUDO_DEBUG_UTIL);

    for (fac = facilities; fac->name != NULL; fac++) {
	if (strcmp(str, fac->name) == 0) {
	    *logfac = fac->num;
	    debug_return_bool(true);
	}
    }
    debug_return_bool(false);
}

const char *
sudo_logfac2str_v1(int num)
{
    const struct strmap *fac;
    debug_decl(sudo_logfac2str, SUDO_DEBUG_UTIL);

    for (fac = facilities; fac->name != NULL; fac++) {
	if (fac->num == num)
	    break;
    }
    debug_return_const_str(fac->name);
}
