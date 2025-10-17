/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

#include "sudo_compat.h"
#include "sudo_dso.h"
#include "sudo_util.h"
#include "sudo_fatal.h"

sudo_dso_public int main(int argc, char *argv[]);

static void
usage(void)
{
    fprintf(stderr, "usage: %s plugin.so symbols_file\n", getprogname());
    exit(EXIT_FAILURE);
}

int
main(int argc, char *argv[])
{
    void *handle, *sym;
    const char *plugin_path;
    const char *symbols_file;
    char *cp, line[LINE_MAX];
    FILE *fp;
    int ntests = 0, errors = 0;

    initprogname(argc > 0 ? argv[0] : "check_symbols");

    if (argc != 3)
	usage();
    plugin_path = argv[1];
    symbols_file = argv[2];

    handle = sudo_dso_load(plugin_path, SUDO_DSO_LAZY|SUDO_DSO_GLOBAL);
    if (handle == NULL) {
	const char *errstr = sudo_dso_strerror();
	sudo_fatalx_nodebug("unable to load %s: %s", plugin_path,
	    errstr ? errstr : "unknown error");
    }

    fp = fopen(symbols_file, "r");
    if (fp == NULL)
	sudo_fatal_nodebug("unable to open %s", symbols_file);

    while (fgets(line, sizeof(line), fp) != NULL) {
	ntests++;
	if ((cp = strchr(line, '\n')) != NULL)
	    *cp = '\0';
	sym = sudo_dso_findsym(handle, line);
	if (sym == NULL) {
	    const char *errstr = sudo_dso_strerror();
	    printf("%s: test %d: unable to resolve symbol %s: %s\n",
		getprogname(), ntests, line, errstr ? errstr : "unknown error");
	    errors++;
	}
    }

    /*
     * Make sure unexported symbols are not available.
     */
    ntests++;
    sym = sudo_dso_findsym(handle, "user_in_group");
    if (sym != NULL) {
	printf("%s: test %d: able to resolve local symbol user_in_group\n",
	    getprogname(), ntests);
	errors++;
    }

    sudo_dso_unload(handle);

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    exit(errors);
}
