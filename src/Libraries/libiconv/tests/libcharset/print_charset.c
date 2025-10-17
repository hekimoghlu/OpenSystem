/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>

#include <libcharset.h>

int
main(int argc, char *argv[])
{

	setlocale(LC_ALL, "");

	if (argc == 1) {
		/*
		 * With no arguments, we just print the current
		 * locale_charset().
		 */
		printf("%s\n", locale_charset());
	} else {
		/*
		 * With one or more arguments, each argument is assumed to be a
		 * CHARSETALIASDIR and we'll try each one.  This is really just
		 * used to demonstrate that the mapping is fixed after the first
		 * call.
		 */
		for (int i = 1; i < argc; i++) {
			if (argv[i][0] == '\0')
				unsetenv("CHARSETALIASDIR");
			else
				setenv("CHARSETALIASDIR", argv[i], 1);
			printf("%s\n", locale_charset());
		}
	}

	return (0);
}
