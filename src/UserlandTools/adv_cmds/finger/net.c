/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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
#if 0
#ifndef lint
static char sccsid[] = "@(#)net.c	8.4 (Berkeley) 4/28/95";
#endif
#endif

#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/param.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <wctype.h>
#include <db.h>
#include <err.h>
#include <netdb.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utmpx.h>
#include <wchar.h>
#include "finger.h"

static void cleanup(int sig);
static int do_protocol(const char *name, const struct addrinfo *ai);
static void trying(const struct addrinfo *ai);

void
netfinger(char *name)
{
	int error, multi;
	char *host;
	struct addrinfo *ai, *ai0;
	static struct addrinfo hint;

	host = strrchr(name, '@');
	if (host == NULL)
		return;
	*host++ = '\0';
	signal(SIGALRM, cleanup);
	alarm(TIME_LIMIT);

	hint.ai_flags = AI_CANONNAME;
	hint.ai_family = family;
	hint.ai_socktype = SOCK_STREAM;

	error = getaddrinfo(host, "finger", &hint, &ai0);
	if (error) {
		warnx("%s: %s", host, gai_strerror(error));
		return;
	}

	multi = (ai0->ai_next) != 0;

	/* ai_canonname may not be filled in if the user specified an IP. */
	if (ai0->ai_canonname == 0)
		printf("[%s]\n", host);
	else
		printf("[%s]\n", ai0->ai_canonname);

	for (ai = ai0; ai != NULL; ai = ai->ai_next) {
		if (multi)
			trying(ai);

		error = do_protocol(name, ai);
		if (!error)
			break;
	}
	alarm(0);
	freeaddrinfo(ai0);
}

static int
do_protocol(const char *name, const struct addrinfo *ai)
{
	int cnt, line_len, s;
	FILE *fp;
	wint_t c, lastc;
	struct iovec iov[3];
	struct msghdr msg;
	static char slash_w[] = "/W ";
	static char neteol[] = "\r\n";

	s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
	if (s < 0) {
		warn("socket(%d, %d, %d)", ai->ai_family, ai->ai_socktype,
		     ai->ai_protocol);
		return -1;
	}

	msg.msg_name = (void *)ai->ai_addr;
	msg.msg_namelen = ai->ai_addrlen;
	msg.msg_iov = iov;
	msg.msg_iovlen = 0;
	msg.msg_control = 0;
	msg.msg_controllen = 0;
	msg.msg_flags = 0;

	/* -l flag for remote fingerd  */
	if (lflag) {
		iov[msg.msg_iovlen].iov_base = slash_w;
		iov[msg.msg_iovlen++].iov_len = 3;
	}
	/* send the name followed by <CR><LF> */
	iov[msg.msg_iovlen].iov_base = strdup(name);
	iov[msg.msg_iovlen++].iov_len = strlen(name);
	iov[msg.msg_iovlen].iov_base = neteol;
	iov[msg.msg_iovlen++].iov_len = 2;

	if (connect(s, ai->ai_addr, ai->ai_addrlen) < 0) {
		warn("connect");
		close(s);
		return -1;
	}

	if (sendmsg(s, &msg, 0) < 0) {
		warn("sendmsg");
		close(s);
		return -1;
	}

	/*
	 * Read from the remote system; once we're connected, we assume some
	 * data.  If none arrives, we hang until the user interrupts.
	 *
	 * If we see a <CR> or a <CR> with the high bit set, treat it as
	 * a newline; if followed by a newline character, only output one
	 * newline.
	 *
	 * Otherwise, all high bits are stripped; if it isn't printable and
	 * it isn't a space, we can simply set the 7th bit.  Every ASCII
	 * character with bit 7 set is printable.
	 */
	lastc = 0;
	if ((fp = fdopen(s, "r")) != NULL) {
		cnt = 0;
		line_len = 0;
		while ((c = getwc(fp)) != EOF) {
			if (++cnt > OUTPUT_MAX) {
				printf("\n\n Output truncated at %d bytes...\n",
					cnt - 1);
				break;
			}
			if (c == 0x0d) {
				if (lastc == '\r')	/* ^M^M - skip dupes */
					continue;
				c = '\n';
				lastc = '\r';
			} else {
				if (!iswprint(c) && !iswspace(c)) {
					c &= 0x7f;
					c |= 0x40;
				}
				if (lastc != '\r' || c != '\n')
					lastc = c;
				else {
					lastc = '\n';
					continue;
				}
			}
			putwchar(c);
			if (c != '\n' && ++line_len > _POSIX2_LINE_MAX) {
				putchar('\\');
				putchar('\n');
				lastc = '\r';
			}
			if (lastc == '\n' || lastc == '\r')
				line_len = 0;
		}
		if (ferror(fp)) {
			/*
			 * Assume that whatever it was set errno...
			 */
			warn("reading from network");
		}
		if (lastc != L'\n')
			putchar('\n');

		fclose(fp);
	}
	return 0;
}

static void
trying(const struct addrinfo *ai)
{
	char buf[NI_MAXHOST];

	if (getnameinfo(ai->ai_addr, ai->ai_addrlen, buf, sizeof buf,
			(char *)0, 0, NI_NUMERICHOST) != 0)
		return;		/* XXX can't happen */

	printf("Trying %s...\n", buf);
}

static void
cleanup(int sig __unused)
{
#define	ERRSTR	"Timed out.\n"
	write(STDERR_FILENO, ERRSTR, sizeof ERRSTR);
	exit(1);
}

