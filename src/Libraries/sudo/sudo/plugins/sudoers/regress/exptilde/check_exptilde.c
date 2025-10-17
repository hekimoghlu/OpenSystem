/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SUDO_ERROR_WRAP 0

#include "sudoers.h"

#include <def_data.c>

struct sudo_user sudo_user;

struct test_data {
    const char *input;
    const char *output;
    const char *user;
    bool result;
} test_data[] = {
    { "foo/bar", NULL, NULL, false },
    { "~root", "/", NULL, true },
    { "~", "/home/millert", "millert", true },
    { "~/foo", "/home/millert/foo", "millert", true },
    { "~millert", "/home/millert", "millert", true },
    { "~millert/bar", "/home/millert/bar", "millert", true },
    { NULL }
};

sudo_dso_public int main(int argc, char *argv[]);

int
main(int argc, char *argv[])
{
    int ntests = 0, errors = 0;
    struct test_data *td;
    struct passwd *pw;
    char *path = NULL;
    bool result;

    initprogname(argc > 0 ? argv[0] : "check_exptilde");

    /* Prime the passwd cache */
    pw = sudo_mkpwent("root", 0, 0, "/", "/bin/sh");
    if (pw == NULL)
	sudo_fatalx("unable to create passwd entry for root");
    sudo_pw_delref(pw);

    pw = sudo_mkpwent("millert", 8036, 20, "/home/millert", "/bin/tcsh");
    if (pw == NULL)
	sudo_fatalx("unable to create passwd entry for millert");
    sudo_pw_delref(pw);

    for (td = test_data; td->input != NULL; td++) {
	ntests++;
	free(path);
	if ((path = strdup(td->input)) == NULL)
	    sudo_fatal(NULL);
	result = expand_tilde(&path, td->user);
	if (result != td->result) {
	    errors++;
	    if (result) {
		sudo_warnx("unexpected success: input %s, output %s", 
		    td->input, path);
	    } else {
		sudo_warnx("unexpected failure: input %s", td->input);
	    }
	    continue;
	}
	if (td->result && strcmp(path, td->output) != 0) {
	    errors++;
	    sudo_warnx("incorrect output for input %s: expected %s, got %s",
		td->input, td->output, path);
	    continue;
	}
    }

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    exit(errors);
}
