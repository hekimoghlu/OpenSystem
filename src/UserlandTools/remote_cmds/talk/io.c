/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include <sys/cdefs.h>

#ifndef __APPLE__
__FBSDID("$FreeBSD$");

#ifndef lint
static const char sccsid[] = "@(#)io.c	8.1 (Berkeley) 6/6/93";
#endif
#endif /* __APPLE__ */

/*
 * This file contains the I/O handling and the exchange of
 * edit characters. This connection itself is established in
 * ctl.c
 */

#include <sys/filio.h>

#include <errno.h>
#include <signal.h>
#include <netdb.h>
#include <poll.h>
#ifdef __APPLE__
#define INFTIM -1
#endif /* __APPLE__ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define _XOPEN_SOURCE_EXTENDED
#include <curses.h>

#include "talk.h"
#include "talk_ctl.h"

extern void	display(xwin_t *, wchar_t *);

volatile sig_atomic_t gotwinch = 0;

/*
 * The routine to do the actual talking
 */
void
talk(void)
{
	struct hostent *hp, *hp2;
	struct pollfd fds[2];
	int nb;
	wchar_t buf[BUFSIZ];
	char **addr, *his_machine_name;
	FILE *sockfp;

	his_machine_name = NULL;
	hp = gethostbyaddr((const char *)&his_machine_addr.s_addr,
	    sizeof(his_machine_addr.s_addr), AF_INET);
	if (hp != NULL) {
		hp2 = gethostbyname(hp->h_name);
		if (hp2 != NULL && hp2->h_addrtype == AF_INET &&
		    hp2->h_length == sizeof(his_machine_addr))
			for (addr = hp2->h_addr_list; *addr != NULL; addr++)
				if (memcmp(*addr, &his_machine_addr,
				    sizeof(his_machine_addr)) == 0) {
					his_machine_name = strdup(hp->h_name);
					break;
				}
	}
	if (his_machine_name == NULL)
		his_machine_name = strdup(inet_ntoa(his_machine_addr));
	snprintf((char *)buf, sizeof(buf), "Connection established with %s@%s.",
	    msg.r_name, his_machine_name);
	free(his_machine_name);
	message((char *)buf);
	write(STDOUT_FILENO, "\007\007\007", 3);
	
	current_line = 0;

	if ((sockfp = fdopen(sockt, "w+")) == NULL)
		p_error("fdopen");

	setvbuf(sockfp, NULL, _IONBF, 0);
	setvbuf(stdin, NULL, _IONBF, 0);

	/*
	 * Wait on both the other process (sockt) and standard input.
	 */
	for (;;) {
		fds[0].fd = fileno(stdin);
		fds[0].events = POLLIN;
		fds[1].fd = sockt;
		fds[1].events = POLLIN;
		nb = poll(fds, 2, INFTIM);
		if (gotwinch) {
			resize_display();
			gotwinch = 0;
		}
		if (nb <= 0) {
			if (errno == EINTR)
				continue;
			/* Panic, we don't know what happened. */
			p_error("Unexpected error from poll");
			quit();
		}
		if (fds[1].revents & POLLIN) {
			wint_t w;

			/* There is data on sockt. */
			w = fgetwc(sockfp);
			if (w == WEOF) {
				message("Connection closed. Exiting");
				quit();
			}
			display(&his_win, &w);
		}
		if (fds[0].revents & POLLIN) {
			wint_t w;

			if ((w = getwchar()) != WEOF) {
				display(&my_win, &w);
				(void )fputwc(w, sockfp);
				(void )fflush(sockfp);
			}
		}
	}
}

/*
 * p_error prints the system error message on the standard location
 * on the screen and then exits. (i.e. a curses version of perror)
 */
void
p_error(const char *string)
{
	wmove(my_win.x_win, current_line, 0);
	wprintw(my_win.x_win, "[%s : %s (%d)]\n",
	    string, strerror(errno), errno);
	wrefresh(my_win.x_win);
	move(LINES-1, 0);
	refresh();
	quit();
}

/*
 * Display string in the standard location
 */
void
message(const char *string)
{
	wmove(my_win.x_win, current_line, 0);
	wprintw(my_win.x_win, "[%s]\n", string);
	if (current_line < my_win.x_nlines - 1)
		current_line++;
	wrefresh(my_win.x_win);
}
