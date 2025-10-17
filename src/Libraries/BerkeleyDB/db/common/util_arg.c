/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#include "db_config.h"

#include "db_int.h"

#if DB_VERSION_MAJOR < 4 || DB_VERSION_MAJOR == 4 && DB_VERSION_MINOR < 5
/*
 * !!!
 * We build this file in old versions of Berkeley DB when we're doing test
 * runs using the test_micro tool.   Without a prototype in place, we get
 * warnings, and there's no simple workaround.
 */
char *strsep();
#endif

/*
 * __db_util_arg --
 *	Convert a string into an argc/argv pair.
 *
 * PUBLIC: int __db_util_arg __P((char *, char *, int *, char ***));
 */
int
__db_util_arg(arg0, str, argcp, argvp)
	char *arg0, *str, ***argvp;
	int *argcp;
{
	int n, ret;
	char **ap, **argv;

#define	MAXARGS	25
	if ((ret =
	    __os_malloc(NULL, (MAXARGS + 1) * sizeof(char **), &argv)) != 0)
		return (ret);

	ap = argv;
	*ap++ = arg0;
	for (n = 1; (*ap = strsep(&str, " \t")) != NULL;)
		if (**ap != '\0') {
			++ap;
			if (++n == MAXARGS)
				break;
		}
	*ap = NULL;

	*argcp = ap - argv;
	*argvp = argv;

	return (0);
}
