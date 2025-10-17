/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

#include <stdlib.h>

#if !HAVE_DECL_ENVIRON
extern char **environ;
#endif

/*
 * putenv --
 *	String points to a string of the form name=value.
 *
 *      Makes the value of the environment variable name equal to
 *      value by altering an existing variable or creating a new one.
 */

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
putenv(const char *string)
{
    int i;
    const char *eq = (const char *)strchr(string, '=');
    int len;

    if (eq == NULL)
	return 1;
    len = eq - string;

    if(environ == NULL) {
	environ = malloc(sizeof(char*));
	if(environ == NULL)
	    return 1;
	environ[0] = NULL;
    }

    for(i = 0; environ[i] != NULL; i++)
	if(strncmp(string, environ[i], len) == 0) {
	    environ[i] = string;
	    return 0;
	}
    environ = realloc(environ, sizeof(char*) * (i + 2));
    if(environ == NULL)
	return 1;
    environ[i]   = string;
    environ[i+1] = NULL;
    return 0;
}
