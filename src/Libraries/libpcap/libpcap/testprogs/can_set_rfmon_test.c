/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#include "varattrs.h"

#ifndef lint
static const char copyright[] _U_ =
    "@(#) Copyright (c) 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 2000\n\
The Regents of the University of California.  All rights reserved.\n";
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include <pcap.h>

#include "pcap/funcattrs.h"

static const char *program_name;

/* Forwards */
static void PCAP_NORETURN error(PCAP_FORMAT_STRING(const char *), ...) PCAP_PRINTFLIKE(1,2);

int
main(int argc, char **argv)
{
	const char *cp;
	pcap_t *pd;
	char ebuf[PCAP_ERRBUF_SIZE];
	int status;

	if ((cp = strrchr(argv[0], '/')) != NULL)
		program_name = cp + 1;
	else
		program_name = argv[0];

	if (argc != 2) {
		fprintf(stderr, "Usage: %s <device>\n", program_name);
		return 2;
	}

	pd = pcap_create(argv[1], ebuf);
	if (pd == NULL)
		error("%s", ebuf);
	status = pcap_can_set_rfmon(pd);
	if (status < 0) {
		if (status == PCAP_ERROR)
			error("%s: pcap_can_set_rfmon failed: %s", argv[1],
			    pcap_geterr(pd));
		else
			error("%s: pcap_can_set_rfmon failed: %s", argv[1],
			    pcap_statustostr(status));
	}
	printf("%s: Monitor mode %s be set\n", argv[1], status ? "can" : "cannot");
	return 0;
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
