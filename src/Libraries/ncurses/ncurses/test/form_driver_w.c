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
/****************************************************************************
 *   Author:  Gaute Hope, 2013                                              *
 ****************************************************************************/

/*
 * $Id: form_driver_w.c,v 1.13 2014/08/02 17:24:55 tom Exp $
 *
 * Test form_driver_w (int, int, wchar_t), a wide char aware
 * replacement of form_driver.
 */

#include <locale.h>

#include <test.priv.h>

#if USE_WIDEC_SUPPORT && USE_LIBFORM && (defined(NCURSES_VERSION_PATCH) && NCURSES_VERSION_PATCH >= 20131207)

#include <form.h>

int
main(void)
{
    FIELD *field[3];
    FORM *my_form;
    bool done = FALSE;

    setlocale(LC_ALL, "");

    /* Initialize curses */
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);

    /* Initialize the fields */
    field[0] = new_field(1, 10, 4, 18, 0, 0);
    field[1] = new_field(1, 10, 6, 18, 0, 0);
    field[2] = NULL;

    /* Set field options */
    set_field_back(field[0], A_UNDERLINE);	/* Print a line for the option  */
    field_opts_off(field[0], O_AUTOSKIP);	/* Don't go to next field when this */
    /* Field is filled up           */
    set_field_back(field[1], A_UNDERLINE);
    field_opts_off(field[1], O_AUTOSKIP);

    /* Create the form and post it */
    my_form = new_form(field);
    post_form(my_form);
    refresh();

    mvprintw(4, 10, "Value 1:");
    mvprintw(6, 10, "Value 2:");
    refresh();

    /* Loop through to get user requests */
    while (!done) {
	wint_t ch;
	int ret = get_wch(&ch);

	mvprintw(8, 10, "Got %d (%#x), type: %s", (int) ch, (int) ch,
		 (ret == KEY_CODE_YES)
		 ? "KEY_CODE_YES"
		 : ((ret == OK)
		    ? "OK"
		    : ((ret == ERR)
		       ? "ERR"
		       : "?")));
	clrtoeol();

	switch (ret) {
	case KEY_CODE_YES:
	    switch (ch) {
	    case KEY_DOWN:
		/* Go to next field */
		form_driver_w(my_form, KEY_CODE_YES, REQ_NEXT_FIELD);
		/* Go to the end of the present buffer */
		/* Leaves nicely at the last character */
		form_driver_w(my_form, KEY_CODE_YES, REQ_END_LINE);
		break;
	    case KEY_UP:
		/* Go to previous field */
		form_driver_w(my_form, KEY_CODE_YES, REQ_PREV_FIELD);
		form_driver_w(my_form, KEY_CODE_YES, REQ_END_LINE);
		break;
	    default:
		break;
	    }
	    break;
	case OK:
	    switch (ch) {
	    case CTRL('D'):
	    case QUIT:
	    case ESCAPE:
		done = TRUE;
		break;
	    default:
		form_driver_w(my_form, OK, (wchar_t) ch);
		break;
	    }
	    break;
	}
    }

    /* Un post form and free the memory */
    unpost_form(my_form);
    free_form(my_form);
    free_field(field[0]);
    free_field(field[1]);

    endwin();
    ExitProgram(EXIT_SUCCESS);
}

#else
int
main(void)
{
    printf("This program requires the wide-ncurses and forms library\n");
    ExitProgram(EXIT_FAILURE);
}
#endif /* USE_WIDEC_SUPPORT */
