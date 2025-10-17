/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
static const char rcsid[] =
    "@(#) $Id: savestr.c,v 1.2 2006/03/09 02:27:11 leres Exp $ (LBL)";
#endif

#include <sys/types.h>

#include <stdio.h>
#include <stdlib.h>

#include "gnuc.h"
#ifdef HAVE_OS_PROTO_H
#include "os-proto.h"
#endif

#include "savestr.h"

/* A replacement for strdup() that cuts down on malloc() overhead */
char *
savestr(register const char *str)
{
	register u_int size;
	register char *p;
	static char *strptr = NULL;
	static u_int strsize = 0;

	size = strlen(str) + 1;
	if (size > strsize) {
		strsize = 1024;
		if (strsize < size)
			strsize = size;
		strptr = (char *)malloc(strsize);
		if (strptr == NULL) {
			fprintf(stderr, "savestr: malloc\n");
			exit(1);
		}
	}
	(void)strcpy(strptr, str);
	p = strptr;
	strptr += size;
	strsize -= size;
	return (p);
}
