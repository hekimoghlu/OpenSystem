/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
#include "top.h"
#include "userinput_order.h"
#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
mode_completion(void *tinst, struct user_input_state *s)
{
	if (!strlen(s->buf)) {
		/* Use the current mode. */
		user_input_set_state(NULL);
		return;
	}

	if (top_prefs_set_mode(s->buf)) {
		char errbuf[60];
		if (-1 == snprintf(errbuf, sizeof(errbuf), "invalid mode: %s\n", s->buf)) {
			user_input_set_error_state("mode error buffer overflow");
			return;
		}
		user_input_set_error_state(errbuf);
		return;
	}

	/*Success*/

	/*
	 * This has an order dependency, and assumes that the
	 * relayout will be lazy.
	 */
	top_relayout_force();
	top_insert(tinst);
	user_input_set_state(NULL);
}

static void
mode_draw(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column)
{
	char display[60];

	if (-1
			== snprintf(display, sizeof(display), "mode [%s]: %s\n", top_prefs_get_mode_string(),
					s->buf))
		return;

	mvwaddstr(win, row, column, display);
}

struct user_input_state top_user_input_mode_state
		= { .offset = 0, .completion = mode_completion, .draw = mode_draw };
