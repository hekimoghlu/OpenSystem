/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pwd.h>

#include "sudoers.h"

/*
 * Expand %h and %u escapes (if present) in the prompt and pass back
 * the dynamically allocated result.
 */
char *
expand_prompt(const char *old_prompt, const char *auth_user)
{
    size_t len, n;
    int subst;
    const char *p;
    char *np, *new_prompt;
    debug_decl(expand_prompt, SUDOERS_DEBUG_AUTH);

    /* How much space do we need to malloc for the prompt? */
    subst = 0;
    for (p = old_prompt, len = strlen(old_prompt); *p != '\0'; p++) {
	if (p[0] =='%') {
	    switch (p[1]) {
		case 'h':
		    p++;
		    len += strlen(user_shost) - 2;
		    subst = 1;
		    break;
		case 'H':
		    p++;
		    len += strlen(user_host) - 2;
		    subst = 1;
		    break;
		case 'p':
		    p++;
		    len += strlen(auth_user) - 2;
		    subst = 1;
		    break;
		case 'u':
		    p++;
		    len += strlen(user_name) - 2;
		    subst = 1;
		    break;
		case 'U':
		    p++;
		    len += strlen(runas_pw->pw_name) - 2;
		    subst = 1;
		    break;
		case '%':
		    p++;
		    len--;
		    subst = 1;
		    break;
		default:
		    break;
	    }
	}
    }

    if ((new_prompt = malloc(++len)) == NULL) {
	sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	debug_return_str(NULL);
    }

    if (subst) {
	for (p = old_prompt, np = new_prompt; *p != '\0'; p++) {
	    if (p[0] =='%') {
		switch (p[1]) {
		    case 'h':
			p++;
			n = strlcpy(np, user_shost, len);
			if (n >= len)
			    goto oflow;
			np += n;
			len -= n;
			continue;
		    case 'H':
			p++;
			n = strlcpy(np, user_host, len);
			if (n >= len)
			    goto oflow;
			np += n;
			len -= n;
			continue;
		    case 'p':
			p++;
			n = strlcpy(np, auth_user, len);
			if (n >= len)
			    goto oflow;
			np += n;
			len -= n;
			continue;
		    case 'u':
			p++;
			n = strlcpy(np, user_name, len);
			if (n >= len)
			    goto oflow;
			np += n;
			len -= n;
			continue;
		    case 'U':
			p++;
			n = strlcpy(np,  runas_pw->pw_name, len);
			if (n >= len)
			    goto oflow;
			np += n;
			len -= n;
			continue;
		    case '%':
			/* convert %% -> % */
			p++;
			break;
		    default:
			/* no conversion */
			break;
		}
	    }
	    if (len < 2)			/* len includes NUL */
		goto oflow;
	    *np++ = *p;
	    len--;
	}
	if (len != 1)
	    goto oflow;
	*np = '\0';
    } else {
	/* Nothing to expand. */
	memcpy(new_prompt, old_prompt, len);	/* len includes NUL */
    }

    debug_return_str(new_prompt);

oflow:
    /* We pre-allocate enough space, so this should never happen. */
    free(new_prompt);
    sudo_warnx(U_("internal error, %s overflow"), __func__);
    debug_return_str(NULL);
}
