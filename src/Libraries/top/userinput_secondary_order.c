/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#include "userinput_secondary_order.h"
#include "preferences.h"
#include <curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
order_completion(void *tinst, struct user_input_state *s)
{
	if (!strlen(s->buf)) {
		/* Use the current order. */
		user_input_set_state(NULL);
		return;
	}

	if (top_prefs_set_secondary_sort(s->buf)) {
		char errbuf[60];
		if (-1 == snprintf(errbuf, sizeof(errbuf), "invalid order: %s\n", s->buf)) {
			user_input_set_error_state("order buffer overflow");
			return;
		}
		user_input_set_error_state(errbuf);
		return;
	}

	user_input_set_state(NULL);
}

static void
order_draw(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column)
{
	char display[60];

	if (-1
			== snprintf(display, sizeof(display), "secondary key [%c%s]: %s\n",
					top_prefs_get_secondary_ascending() ? '+' : '-',
					top_prefs_get_secondary_sort_string(), s->buf))
		return;

	mvwaddstr(win, row, column, display);
}

struct user_input_state top_user_input_secondary_order_state
		= { .offset = 0, .completion = order_completion, .draw = order_draw };
