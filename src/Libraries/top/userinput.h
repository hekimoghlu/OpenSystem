/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#ifndef USERINPUT_H
#define USERINPUT_H

#include <stdbool.h>
#include <curses.h>

struct user_input_state {
	char buf[60];
	int offset;
	/* This is called when Return is pressed. */
	void (*completion)(void *tinst, struct user_input_state *s);

	/* This is called to draw the current input text, and a prompt. */
	void (*draw)(void *tinst, struct user_input_state *s, WINDOW *win, int row, int column);

	int misc; /* A variable for misc things that each state may use. */
};

bool user_input_process(void *tinst);
void user_input_set_position(int r, int c);
void user_input_draw(void *tinst, WINDOW *win);
void user_input_set_error_state(const char *err);
void user_input_set_state(struct user_input_state *state);

#endif
