/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
 * $Id: key_names.c,v 1.5 2014/08/02 17:24:55 tom Exp $
 */

#include <test.priv.h>

#if USE_WIDEC_SUPPORT

static void
usage(void)
{
    fprintf(stderr, "Usage: key_names [-m] [-s]\n");
    ExitProgram(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
    int n;

    bool do_setup = FALSE;
    bool do_meta = FALSE;

    setlocale(LC_ALL, "");

    while ((n = getopt(argc, argv, "ms")) != -1) {
	switch (n) {
	case 'm':
	    do_meta = TRUE;
	    break;
	case 's':
	    do_setup = TRUE;
	    break;
	default:
	    usage();
	    /* NOTREACHED */
	}
    }

    if (do_setup) {
	/*
	 * Get the terminfo entry into memory, and tell ncurses that we want to
	 * use function keys.  That will make it add any user-defined keys that
	 * appear in the terminfo.
	 */
	newterm(getenv("TERM"), stderr, stdin);
	keypad(stdscr, TRUE);
	if (do_meta)
	    meta(stdscr, TRUE);
	endwin();
    }
    for (n = -1; n < KEY_MAX + 512; n++) {
	const char *result = key_name((wchar_t) n);
	if (result != 0)
	    printf("%d(%5o):%s\n", n, n, result);
    }
    ExitProgram(EXIT_SUCCESS);
}
#else
int
main(void)
{
    printf("This program requires the wide-ncurses library\n");
    ExitProgram(EXIT_FAILURE);
}
#endif
