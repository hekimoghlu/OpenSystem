/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
#include <sys/types.h>

#include <err.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "extern.h"

void
eofmsg(const char *file)
{
	if (!sflag)
		warnx("EOF on %s", file);
}

void
diffmsg(const char *file1, const char *file2, off_t byte, off_t line,
    int b1, int b2)
{
	if (sflag) {
		/* nothing */
	} else if (bflag) {
		(void)printf("%s %s differ: char %lld, line %lld is %3o %c %3o %c\n",
		    file1, file2, (long long)byte, (long long)line, b1, b1,
		    b2, b2);
	} else {
		(void)printf("%s %s differ: char %lld, line %lld\n",
		    file1, file2, (long long)byte, (long long)line);
	}
}
