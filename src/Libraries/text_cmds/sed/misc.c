/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
__FBSDID("$FreeBSD$");

#ifndef lint
static const char sccsid[] = "@(#)misc.c	8.1 (Berkeley) 6/6/93";
#endif

#include <sys/types.h>

#include <err.h>
#include <limits.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defs.h"
#include "extern.h"

/*
 * Return a string for a regular expression error passed.  This is overkill,
 * because of the silly semantics of regerror (we can never know the size of
 * the buffer).
 */
char *
strregerror(int errcode, regex_t *preg)
{
	static char *oe;
	size_t s;

	if (oe != NULL)
		free(oe);
	s = regerror(errcode, preg, NULL, 0);
	if ((oe = malloc(s)) == NULL)
		err(1, "malloc");
	(void)regerror(errcode, preg, oe, s);
	return (oe);
}
