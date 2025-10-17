/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#ifndef lint
static const char copyright[] =
    "@(#) Copyright (c) 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 2000\n\
The Regents of the University of California.  All rights reserved.\n";
#endif

#include <pcap.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/select.h>
#include <poll.h>

char *program_name;

/* Forwards */
static void countme(u_char *, const struct pcap_pkthdr *, const u_char *);
static void usage(void) __attribute__((noreturn));
static void error(const char *, ...);
static void warning(const char *, ...);
static char *copy_argv(char **);

static pcap_t *pd;

extern int optind;
extern int opterr;
extern char *optarg;

int
main(int argc, char **argv)
{
	register int op;
	bpf_u_int32 localnet, netmask;
	register char *cp, *cmdbuf, *device;
	struct bpf_program fcode;
	char ebuf[PCAP_ERRBUF_SIZE];
	int status;
	int packet_count;

	device = NULL;
	if ((cp = strrchr(argv[0], '/')) != NULL)
		program_name = cp + 1;
	else
		program_name = argv[0];

	opterr = 0;
	while ((op = getopt(argc, argv, "i:")) != -1) {
		switch (op) {

		case 'i':
			device = optarg;
			break;

		default:
			usage();
			/* NOTREACHED */
		}
	}

	if (device == NULL) {
		device = pcap_lookupdev(ebuf);
		if (device == NULL)
			error("%s", ebuf);
	}
	*ebuf = '\0';
	pd = pcap_open_live(device, 65535, 0, 1000, ebuf);
	if (pd == NULL)
		error("%s", ebuf);
	else if (*ebuf)
		warning("%s", ebuf);
	if (pcap_lookupnet(device, &localnet, &netmask, ebuf) < 0) {
		localnet = 0;
		netmask = 0;
		warning("%s", ebuf);
	}
	cmdbuf = copy_argv(&argv[optind]);

	if (pcap_compile(pd, &fcode, cmdbuf, 1, netmask) < 0)
		error("%s", pcap_geterr(pd));

	if (pcap_setfilter(pd, &fcode) < 0)
		error("%s", pcap_geterr(pd));
	if (pcap_setnonblock(pd, 1, ebuf) == -1)
		error("pcap_setnonblock failed: %s", ebuf);
	printf("Listening on %s\n", device);
	for (;;) {
		packet_count = 0;
		status = pcap_dispatch(pd, -1, countme,
		    (u_char *)&packet_count);
		if (status < 0)
			break;
		if (status != 0) {
			printf("%d packets seen, %d packets counted after pcap_dispatch returns\n",
			    status, packet_count);
		}
	}
	if (status == -2) {
		/*
		 * We got interrupted, so perhaps we didn't
		 * manage to finish a line we were printing.
		 * Print an extra newline, just in case.
		 */
		putchar('\n');
	}
	(void)fflush(stdout);
	if (status == -1) {
		/*
		 * Error.  Report it.
		 */
		(void)fprintf(stderr, "%s: pcap_loop: %s\n",
		    program_name, pcap_geterr(pd));
	}
	pcap_close(pd);
	exit(status == -1 ? 1 : 0);
}

static void
countme(u_char *user, const struct pcap_pkthdr *h, const u_char *sp)
{
	int *counterp = (int *)user;

	(*counterp)++;
}

static void
usage(void)
{
	(void)fprintf(stderr, "Usage: %s [ -sptn ] [ -i interface ] [expression]\n",
	    program_name);
	exit(1);
}

/* VARARGS */
static void
error(const char *fmt, ...)
{
	va_list ap;

	(void)fprintf(stderr, "%s: ", program_name);
	va_start(ap, fmt);
	(void)vfprintf(stderr, fmt, ap);
	va_end(ap);
	if (*fmt) {
		fmt += strlen(fmt);
		if (fmt[-1] != '\n')
			(void)fputc('\n', stderr);
	}
	exit(1);
	/* NOTREACHED */
}

/* VARARGS */
static void
warning(const char *fmt, ...)
{
	va_list ap;

	(void)fprintf(stderr, "%s: WARNING: ", program_name);
	va_start(ap, fmt);
	(void)vfprintf(stderr, fmt, ap);
	va_end(ap);
	if (*fmt) {
		fmt += strlen(fmt);
		if (fmt[-1] != '\n')
			(void)fputc('\n', stderr);
	}
}

/*
 * Copy arg vector into a new buffer, concatenating arguments with spaces.
 */
static char *
copy_argv(register char **argv)
{
	register char **p;
	register u_int len = 0;
	char *buf;
	char *src, *dst;

	p = argv;
	if (*p == 0)
		return 0;

	while (*p)
		len += strlen(*p++) + 1;

	buf = (char *)malloc(len);
	if (buf == NULL)
		error("copy_argv: malloc");

	p = argv;
	dst = buf;
	while ((src = *p++) != NULL) {
		while ((*dst++ = *src++) != '\0')
			;
		dst[-1] = ' ';
	}
	dst[-1] = '\0';

	return buf;
}
