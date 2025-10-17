/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
/* gleaned from a web-search, shows a bug combining scanw and implicit scroll.
 * Date:  1997/03/17
 * From:  bayern@morpheus.cis.yale.edu
 *
 * $Id: testscanw.c,v 1.11 2006/04/01 19:08:03 tom Exp $
 */
#include <test.priv.h>

int
main(int argc, char *argv[])
{
    long badanswer = 1;
    long *response = &badanswer;

    setlocale(LC_ALL, "");

    initscr();
    scrollok(stdscr, TRUE);
    idlok(stdscr, TRUE);
    echo();

#if 0
    trace(TRACE_UPDATE | TRACE_CALLS);
#endif
    while (argc > 1) {
	if (isdigit(UChar(*argv[1])))
	    move(atoi(argv[1]), 0);
	else if (!strcmp(argv[1], "-k"))
	    keypad(stdscr, TRUE);
	argc--, argv++;
    }

    while (badanswer) {
	printw("Enter a number (0 to quit):\n");
	printw("--> ");
	scanw("%20ld", response);	/* yes, it's a pointer */
    }
    endwin();
    ExitProgram(EXIT_SUCCESS);
}
