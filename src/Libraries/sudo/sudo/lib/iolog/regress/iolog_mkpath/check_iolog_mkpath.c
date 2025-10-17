/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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

#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define SUDO_ERROR_WRAP 0

#include "sudo_compat.h"
#include "sudo_util.h"
#include "sudo_fatal.h"
#include "sudo_iolog.h"

sudo_dso_public int main(int argc, char *argv[]);

static const char *test_paths[] = {
    "testdir/a/b/c/user",		/* create new */
    "testdir/a/b/c/user",		/* open existing */
    "testdir/a/b/c/user.XXXXXX",	/* mkdtemp new */
    NULL
};

static void
test_iolog_mkpath(const char *testdir, int *ntests, int *nerrors)
{
    const char **tp;
    char *path;

    iolog_set_owner(geteuid(), getegid());

    for (tp = test_paths; *tp != NULL; tp++) {
	if (asprintf(&path, "%s/%s", testdir, *tp) == -1)
	    sudo_fatalx("unable to allocate memory");

	(*ntests)++;
	if (!iolog_mkpath(path)) {
	    sudo_warnx("unable to mkpath %s", path);
	    (*nerrors)++;
	}
	free(path);
    }
}

int
main(int argc, char *argv[])
{
    char testdir[] = "mkpath.XXXXXX";
    const char *rmargs[] = { "rm", "-rf", NULL, NULL };
    int ch, status, ntests = 0, errors = 0;

    initprogname(argc > 0 ? argv[0] : "check_iolog_mkpath");

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    /* ignore */
	    break;
	default:
	    fprintf(stderr, "usage: %s [-v]\n", getprogname());
	    return EXIT_FAILURE;
	}
    }
    argc -= optind;
    argv += optind;

    if (mkdtemp(testdir) == NULL)
	sudo_fatal("unable to create test dir");
    rmargs[2] = testdir;

    test_iolog_mkpath(testdir, &ntests, &errors);

    if (ntests != 0) {
	printf("iolog_mkpath: %d test%s run, %d errors, %d%% success rate\n",
	    ntests, ntests == 1 ? "" : "s", errors,
	    (ntests - errors) * 100 / ntests);
    }

    /* Clean up (avoid running via shell) */
    switch (fork()) {
    case -1:
	sudo_warn("fork");
	_exit(1);
    case 0:
	execvp("rm", (char **)rmargs);
	_exit(1);
    default:
	wait(&status);
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
	    errors++;
	break;
    }

    return errors;
}
