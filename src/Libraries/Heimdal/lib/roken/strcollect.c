/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#include <config.h>

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "roken.h"

enum { initial = 10, increment = 5 };

static char **
sub (char **argv, int i, int argc, va_list *ap)
{
    do {
	if(i == argc) {
	    /* realloc argv */
	    char **tmp = realloc(argv, (argc + increment) * sizeof(*argv));
	    if(tmp == NULL) {
		free(argv);
		errno = ENOMEM;
		return NULL;
	    }
	    argv  = tmp;
	    argc += increment;
	}
	argv[i++] = va_arg(*ap, char*);
    } while(argv[i - 1] != NULL);
    return argv;
}

/*
 * return a malloced vector of pointers to the strings in `ap'
 * terminated by NULL.
 */

ROKEN_LIB_FUNCTION char ** ROKEN_LIB_CALL
vstrcollect(va_list *ap)
{
    return sub (NULL, 0, 0, ap);
}

/*
 *
 */

ROKEN_LIB_FUNCTION char ** ROKEN_LIB_CALL
strcollect(char *first, ...)
{
    va_list ap;
    char **ret = malloc (initial * sizeof(char *));

    if (ret == NULL)
	return ret;

    ret[0] = first;
    va_start(ap, first);
    ret = sub (ret, 1, initial, &ap);
    va_end(ap);
    return ret;
}
