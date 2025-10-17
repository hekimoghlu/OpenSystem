/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#include "userinput_user.h"
#include "preferences.h"
#include <curses.h>
#include <pwd.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
user_completion(void *tinst, struct user_input_state *s)
{
	struct passwd *pwd;

	if (!strlen(s->buf)) {
		/* The user entered an empty string, so they don't want a user. */
		top_prefs_set_user(s->buf);
		user_input_set_state(NULL);
		return;
	}

	pwd = getpwnam(s->buf);

	if (NULL == pwd) {
		user_input_set_error_state("invalid user");
		endpwent();
		return;
	}

	top_prefs_set_user(s->buf);
	top_prefs_set_user_uid(pwd->pw_uid);

	endpwent();

	user_input_set_state(NULL);
}

static void
user_draw(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column)
{
	const char *curuser = top_prefs_get_user();

	char display[60];

	if (-1 == snprintf(display, sizeof(display), "user [%s]: %s\n", curuser ? curuser : "", s->buf))
		return;

	mvwaddstr(win, row, column, display);
}

struct user_input_state top_user_input_user_state
		= { .offset = 0, .completion = user_completion, .draw = user_draw };
