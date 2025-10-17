/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif /* HAVE_STDBOOL_H */

#define DEFAULT_TEXT_DOMAIN	"sudoers"

#include "sudo_compat.h"
#include "sudo_eventlog.h"
#include "sudo_fatal.h"
#include "sudo_gettext.h"
#include "sudoers_debug.h"

#include "defaults.h"
#include "logging.h"

static int current_locale = SUDOERS_LOCALE_USER;
static char *user_locale;
static char *sudoers_locale;

int
sudoers_getlocale(void)
{
    debug_decl(sudoers_getlocale, SUDOERS_DEBUG_UTIL);
    debug_return_int(current_locale);
}

bool
sudoers_initlocale(const char *ulocale, const char *slocale)
{
    debug_decl(sudoers_initlocale, SUDOERS_DEBUG_UTIL);

    if (ulocale != NULL) {
	free(user_locale);
	if ((user_locale = strdup(ulocale)) == NULL)
	    debug_return_bool(false);
    }
    if (slocale != NULL) {
	free(sudoers_locale);
	if ((sudoers_locale = strdup(slocale)) == NULL)
	    debug_return_bool(false);
    }
    sudo_debug_printf(SUDO_DEBUG_INFO, "%s: user locale %s, sudoers locale %s",
	__func__, user_locale, sudoers_locale);
    debug_return_bool(true);
}

/*
 * Set locale to user or sudoers value.
 * Returns true on success and false on failure,
 * If prev_locale is non-NULL it will be filled in with the
 * old SUDOERS_LOCALE_* value.
 */
bool
sudoers_setlocale(int locale_type, int *prev_locale)
{
    char *res = NULL;
    debug_decl(sudoers_setlocale, SUDOERS_DEBUG_UTIL);

    switch (locale_type) {
	case SUDOERS_LOCALE_USER:
	    if (prev_locale)
		*prev_locale = current_locale;
	    if (current_locale != SUDOERS_LOCALE_USER) {
		current_locale = SUDOERS_LOCALE_USER;
		sudo_debug_printf(SUDO_DEBUG_DEBUG,
		    "%s: setting locale to %s (user)", __func__,
		    user_locale ? user_locale : "");
		res = setlocale(LC_ALL, user_locale ? user_locale : "");
		if (res != NULL && user_locale == NULL) {
		    user_locale = setlocale(LC_ALL, NULL);
		    if (user_locale != NULL)
			user_locale = strdup(user_locale);
		    if (user_locale == NULL)
			res = NULL;
		}
	    }
	    break;
	case SUDOERS_LOCALE_SUDOERS:
	    if (prev_locale)
		*prev_locale = current_locale;
	    if (current_locale != SUDOERS_LOCALE_SUDOERS) {
		current_locale = SUDOERS_LOCALE_SUDOERS;
		sudo_debug_printf(SUDO_DEBUG_DEBUG,
		    "%s: setting locale to %s (sudoers)", __func__,
		    sudoers_locale ? sudoers_locale : "C");
		res = setlocale(LC_ALL, sudoers_locale ? sudoers_locale : "C");
		if (res == NULL && sudoers_locale != NULL) {
		    if (strcmp(sudoers_locale, "C") != 0) {
			free(sudoers_locale);
			sudoers_locale = strdup("C");
			if (sudoers_locale != NULL)
			    res = setlocale(LC_ALL, "C");
		    }
		}
	    }
	    break;
    }
    debug_return_bool(res ? true : false);
}

bool
sudoers_warn_setlocale(bool restore, int *cookie)
{
    debug_decl(sudoers_warn_setlocale, SUDOERS_DEBUG_UTIL);

    if (restore)
	debug_return_bool(sudoers_setlocale(*cookie, NULL));
    debug_return_bool(sudoers_setlocale(SUDOERS_LOCALE_USER, cookie));
}

/*
 * Callback for sudoers_locale sudoers setting.
 */
bool
sudoers_locale_callback(const char *file, int line, int column,
    const union sudo_defs_val *sd_un, int op)
{
    debug_decl(sudoers_locale_callback, SUDOERS_DEBUG_UTIL);

    if (sudoers_initlocale(NULL, sd_un->str)) {
	if (setlocale(LC_ALL, sd_un->str) != NULL)
	    debug_return_bool(true);
    }
    debug_return_bool(false);
}
