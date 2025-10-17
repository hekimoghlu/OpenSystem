/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
/* $Id: gxbn_test.c,v 1.16 2007/06/19 23:46:59 tbox Exp $ */

/*! \file */
#include <config.h>

#include <stdio.h>

#include <isc/net.h>
#include <isc/print.h>

#include <lwres/netdb.h>

static void
print_he(struct hostent *he, int error, const char *fun, const char *name) {
	char **c;
	int i;

	if (he != NULL) {
		 printf("%s(%s):\n", fun, name);
		 printf("\tname = %s\n", he->h_name);
		 printf("\taddrtype = %d\n", he->h_addrtype);
		 printf("\tlength = %d\n", he->h_length);
		 c = he->h_aliases;
		 i = 1;
		 while (*c != NULL) {
			printf("\talias[%d] = %s\n", i, *c);
			i++;
			c++;
		 }
		 c = he->h_addr_list;
		 i = 1;
		 while (*c != NULL) {
			char buf[128];
			inet_ntop(he->h_addrtype, *c, buf, sizeof(buf));
			printf("\taddress[%d] = %s\n", i, buf);
			c++;
			i++;
		}
	} else {
		printf("%s(%s): error = %d (%s)\n", fun, name, error,
		       hstrerror(error));
	}
}

int
main(int argc, char **argv) {
	struct hostent *he;
	int error;

	(void)argc;

	while (argv[1] != NULL) {
		he = gethostbyname(argv[1]);
		print_he(he, h_errno, "gethostbyname", argv[1]);

		he = getipnodebyname(argv[1], AF_INET6, AI_DEFAULT|AI_ALL,
				     &error);
		print_he(he, error, "getipnodebyname", argv[1]);
		if (he != NULL)
			freehostent(he);

		he = getipnodebyname(argv[1], AF_INET6, AI_DEFAULT,
				     &error);
		print_he(he, error, "getipnodebyname", argv[1]);
		if (he != NULL)
			freehostent(he);
		argv++;
	}
	return (0);
}
