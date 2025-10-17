/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "preferences.h"
#include "userinput.h"
#include <ctype.h>
#include <curses.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static void
sleep_completion(void *tinst, struct user_input_state *s)
{
	int delay = 0;
	char *p;
	bool got_digit = false;

	if (!strlen(s->buf)) {
		/* Use the current default. */
		user_input_set_state(NULL);
		return;
	}

	for (p = s->buf; *p; ++p) {
		if (isdigit(*p)) {
			got_digit = true;
		} else {
			user_input_set_error_state("not a valid sleep delay");
			return;
		}
	}

	if (!strlen(s->buf) || !got_digit) {
		user_input_set_error_state("not a valid sleep delay");
		return;
	}

	delay = atoi(s->buf);

	if (delay < 0) {
		user_input_set_error_state("delay is negative");
		return;
	}

	top_prefs_set_sleep(delay);

	user_input_set_state(NULL);
}

static void
sleep_draw(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column)
{
	char display[60];

	if (-1
			== snprintf(display, sizeof(display), "update interval[%d]: %s\n",
					top_prefs_get_sleep(), s->buf))
		return;

	mvwaddstr(win, row, column, display);
}

struct user_input_state top_user_input_sleep_state
		= { .offset = 0, .completion = sleep_completion, .draw = sleep_draw };
