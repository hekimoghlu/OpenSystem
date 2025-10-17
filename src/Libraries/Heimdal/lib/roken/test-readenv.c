/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

#include "roken.h"
#include "test-mem.h"

char *s1 = "VAR1=VAL1#comment\n\
VAR2=VAL2 VAL2 #comment\n\
#this another comment\n\
\n\
VAR3=FOO";

char *s2 = "VAR1=ENV2\n\
";

static void
make_file(char *tmpl, size_t l)
{
    int fd;
    strlcpy(tmpl, "env.XXXXXX", l);
    fd = mkstemp(tmpl);
    if(fd < 0)
	err(1, "mkstemp");
    close(fd);
}

static void
write_file(const char *fn, const char *s)
{
    FILE *f;
    f = fopen(fn, "w");
    if(f == NULL) {
	unlink(fn);
	err(1, "fopen");
    }
    if(fwrite(s, 1, strlen(s), f) != strlen(s))
	err(1, "short write");
    if(fclose(f) != 0) {
	unlink(fn);
	err(1, "fclose");
    }
}

int
main(int argc, char **argv)
{
    char **env = NULL;
    int count = 0;
    char fn[MAXPATHLEN];
    int error = 0;

    make_file(fn, sizeof(fn));

    write_file(fn, s1);
    count = read_environment(fn, &env);
    if(count != 3) {
	warnx("test 1: variable count %d != 3", count);
	error++;
    }

    write_file(fn, s2);
    count = read_environment(fn, &env);
    if(count != 1) {
	warnx("test 2: variable count %d != 1", count);
	error++;
    }

    unlink(fn);
    count = read_environment(fn, &env);
    if(count != 0) {
	warnx("test 3: variable count %d != 0", count);
	error++;
    }
    for(count = 0; env && env[count]; count++);
    if(count != 3) {
	warnx("total variable count %d != 3", count);
	error++;
    }
    free_environment(env);


    return error;
}
