/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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

#include <sys/types.h>			/* for id_t */

#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <ctype.h>
#include <errno.h>
#include <limits.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_gettext.h"
#include "sudo_util.h"

/*
 * Make sure that the ID ends with a valid separator char.
 */
static bool
valid_separator(const char *p, const char *ep, const char *sep)
{
    bool valid = false;

    if (ep != p) {
	/* check for valid separator (including '\0') */
	if (sep == NULL)
	    sep = "";
	do {
	    if (*ep == *sep)
		valid = true;
	} while (*sep++ != '\0');
    }
    return valid;
}

/*
 * Parse a uid/gid in string form.
 * If sep is non-NULL, it contains valid separator characters (e.g. comma, space)
 * If endp is non-NULL it is set to the next char after the ID.
 * On success, returns the parsed ID and clears errstr.
 * On error, returns 0 and sets errstr.
 */
id_t
sudo_strtoidx_v1(const char *p, const char *sep, char **endp, const char **errstrp)
{
    const char *errstr;
    char *ep;
    id_t ret;
    debug_decl(sudo_strtoid, SUDO_DEBUG_UTIL);

    ret = sudo_strtonumx(p, INT_MIN, UINT_MAX, &ep, &errstr);
    if (errstr == NULL) {
	/*
	 * Disallow id -1 (UINT_MAX), which means "no change"
	 * and check for a valid separator (if specified).
	 */
	if (ret == (id_t)-1 || ret == (id_t)UINT_MAX || !valid_separator(p, ep, sep)) {
	    errstr = N_("invalid value");
	    errno = EINVAL;
	    ret = 0;
	}
    }
    if (errstrp != NULL)
	*errstrp = errstr;
    if (endp != NULL)
	*endp = ep;
    debug_return_id_t(ret);
}

/* Backward compatibility */
id_t
sudo_strtoid_v1(const char *p, const char *sep, char **endp, const char **errstrp)
{
    return sudo_strtoidx_v1(p, sep, endp, errstrp);
}

/* Simplified interface */
id_t
sudo_strtoid_v2(const char *p, const char **errstrp)
{
    return sudo_strtoidx_v1(p, NULL, NULL, errstrp);
}
