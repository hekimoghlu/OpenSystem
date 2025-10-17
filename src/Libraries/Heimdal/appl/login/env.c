/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#include "login_locl.h"
RCSID("$Id$");

/*
 * the environment we will send to execle and the shell.
 */

char **env;
int num_env;

void
extend_env(char *str)
{
    env = realloc(env, (num_env + 1) * sizeof(*env));
    if(env == NULL)
	errx(1, "Out of memory!");
    env[num_env++] = str;
}

void
add_env(const char *var, const char *value)
{
    int i;
    char *str;
    asprintf(&str, "%s=%s", var, value);
    if(str == NULL)
	errx(1, "Out of memory!");
    for(i = 0; i < num_env; i++)
	if(strncmp(env[i], var, strlen(var)) == 0 &&
	   env[i][strlen(var)] == '='){
	    free(env[i]);
	    env[i] = str;
	    return;
	}

    extend_env(str);
}

#if !HAVE_DECL_ENVIRON
extern char **environ;
#endif


void
copy_env(void)
{
    char **p;
    for(p = environ; *p; p++)
	extend_env(*p);
}

void
login_read_env(const char *file)
{
    char **newenv;
    char *p;
    int i, j;

    newenv = NULL;
    i = read_environment(file, &newenv);
    for (j = 0; j < i; j++) {
	p = strchr(newenv[j], '=');
	if (p == NULL)
	    errx(1, "%s: missing = in string %s",
		 file, newenv[j]);
	*p++ = 0;
	add_env(newenv[j], p);
	*--p = '=';
	free(newenv[j]);
    }
    free(newenv);
}
