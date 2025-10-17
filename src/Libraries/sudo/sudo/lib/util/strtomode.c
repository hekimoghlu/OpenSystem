/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

#include <sys/stat.h>

#include <stdlib.h>
#include <errno.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_gettext.h"
#include "sudo_util.h"

/*
 * Parse an octal file mode in the range [0, 0777].
 * On success, returns the parsed mode and clears errstr.
 * On error, returns 0 and sets errstr.
 */
int
sudo_strtomode_v1(const char *cp, const char **errstr)
{
    char *ep;
    long lval;
    debug_decl(sudo_strtomode, SUDO_DEBUG_UTIL);

    errno = 0;
    lval = strtol(cp, &ep, 8);
    if (ep == cp || *ep != '\0') {
	if (errstr != NULL)
	    *errstr = N_("invalid value");
	errno = EINVAL;
	debug_return_int(0);
    }
    if (lval < 0 || lval > ACCESSPERMS) {
	if (errstr != NULL)
	    *errstr = lval < 0 ? N_("value too small") : N_("value too large");
	errno = ERANGE;
	debug_return_int(0);
    }
    if (errstr != NULL)
	*errstr = NULL;
    debug_return_int((int)lval);
}
