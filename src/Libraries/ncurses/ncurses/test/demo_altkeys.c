/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
 * $Id: demo_altkeys.c,v 1.9 2010/11/14 00:59:26 tom Exp $
 *
 * Demonstrate the define_key() function.
 * Thomas Dickey - 2005/10/22
 */

#include <test.priv.h>

#if defined(NCURSES_VERSION) && NCURSES_EXT_FUNCS

#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif

#define MY_LOGFILE "demo_altkeys.log"
#define MY_KEYS (KEY_MAX + 1)

/*
 * Log the most recently-written line to our logfile
 */
static void
log_last_line(WINDOW *win)
{
    FILE *fp;
    int y, x, n;
    char temp[256];

    if ((fp = fopen(MY_LOGFILE, "a")) != 0) {
	int need = sizeof(temp) - 1;
	if (need > COLS)
	    need = COLS;
	getyx(win, y, x);
	wmove(win, y - 1, 0);
	n = winnstr(win, temp, need);
	while (n-- > 0) {
	    if (isspace(UChar(temp[n])))
		temp[n] = '\0';
	    else
		break;
	}
	wmove(win, y, x);
	fprintf(fp, "%s\n", temp);
	fclose(fp);
    }
}

int
main(int argc GCC_UNUSED, char *argv[]GCC_UNUSED)
{
    int n;
    int ch;
#if HAVE_GETTIMEOFDAY
    int secs, msecs;
    struct timeval current, previous;
#endif

    unlink(MY_LOGFILE);

    newterm(0, stdout, stdin);
    (void) cbreak();		/* take input chars one at a time, no wait for \n */
    (void) noecho();		/* don't echo input */

    scrollok(stdscr, TRUE);
    keypad(stdscr, TRUE);
    move(0, 0);

    /* we do the define_key() calls after keypad(), since the first call to
     * keypad() initializes the corresponding data.
     */
    for (n = 0; n < 255; ++n) {
	char temp[10];
	sprintf(temp, "\033%c", n);
	define_key(temp, n + MY_KEYS);
    }
    for (n = KEY_MIN; n < KEY_MAX; ++n) {
	char *value;
	if ((value = keybound(n, 0)) != 0) {
	    char *temp = typeMalloc(char, strlen(value) + 2);
	    sprintf(temp, "\033%s", value);
	    define_key(temp, n + MY_KEYS);
	    free(temp);
	    free(value);
	}
    }

#if HAVE_GETTIMEOFDAY
    gettimeofday(&previous, 0);
#endif

    while ((ch = getch()) != ERR) {
	bool escaped = (ch >= MY_KEYS);
	const char *name = keyname(escaped ? (ch - MY_KEYS) : ch);

#if HAVE_GETTIMEOFDAY
	gettimeofday(&current, 0);
	secs = (int) (current.tv_sec - previous.tv_sec);
	msecs = (int) ((current.tv_usec - previous.tv_usec) / 1000);
	if (msecs < 0) {
	    msecs += 1000;
	    --secs;
	}
	if (msecs >= 1000) {
	    secs += msecs / 1000;
	    msecs %= 1000;
	}
	printw("%6d.%03d ", secs, msecs);
	previous = current;
#endif
	printw("Keycode %d, name %s%s\n",
	       ch,
	       escaped ? "ESC-" : "",
	       name != 0 ? name : "<null>");
	log_last_line(stdscr);
	clrtoeol();
	if (ch == 'q')
	    break;
    }
    endwin();
    ExitProgram(EXIT_SUCCESS);
}
#else
int
main(void)
{
    printf("This program requires the ncurses library\n");
    ExitProgram(EXIT_FAILURE);
}
#endif
