/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

#include <pcap.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "pcap/funcattrs.h"

/* Forwards */
static void PCAP_NORETURN error(PCAP_FORMAT_STRING(const char *), ...) PCAP_PRINTFLIKE(1,2);

int
main(void)
{
	char ebuf[PCAP_ERRBUF_SIZE];
	pcap_t *pd;
	int status = 0;

	pd = pcap_open_live("lo0", 65535, 0, 1000, ebuf);
	if (pd == NULL) {
		pd = pcap_open_live("lo", 65535, 0, 1000, ebuf);
		if (pd == NULL) {
			error("Neither lo0 nor lo could be opened: %s",
			    ebuf);
		}
	}
	status = pcap_activate(pd);
	if (status != PCAP_ERROR_ACTIVATED) {
		if (status == 0)
			error("pcap_activate() of opened pcap_t succeeded");
		else if (status == PCAP_ERROR)
			error("pcap_activate() of opened pcap_t failed with %s, not PCAP_ERROR_ACTIVATED",
			    pcap_geterr(pd));
		else
			error("pcap_activate() of opened pcap_t failed with %s, not PCAP_ERROR_ACTIVATED",
			    pcap_statustostr(status));
	}
	return 0;
}

/* VARARGS */
static void
error(const char *fmt, ...)
{
	va_list ap;

	(void)fprintf(stderr, "reactivatetest: ");
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
