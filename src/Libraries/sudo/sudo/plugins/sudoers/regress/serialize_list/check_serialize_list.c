/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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

#define SUDO_ERROR_WRAP 0

#include "sudoers.h"

sudo_dso_public int main(int argc, char *argv[]);

static void
test_serialize_list(int *ntests_out, int *errors_out)
{
    int ntests = *ntests_out;
    int errors = *errors_out;
    const char *expected = "myvar=a value with spaces,this\\,and\\,that,\\,";
    struct list_members members = SLIST_HEAD_INITIALIZER(members);
    struct list_member lm1, lm2, lm3;
    char *result;

    lm1.value = (char *)"a value with spaces";
    lm2.value = (char *)"this,and,that";
    lm3.value = (char *)",";
    SLIST_INSERT_HEAD(&members, &lm3, entries);
    SLIST_INSERT_HEAD(&members, &lm2, entries);
    SLIST_INSERT_HEAD(&members, &lm1, entries);

    ntests++;
    result = serialize_list("myvar", &members);
    if (result == NULL) {
	sudo_warnx("serialize_list returns NULL");
	++errors;
	goto done;
    }
    ntests++;
    if (strcmp(result, expected) != 0) {
	sudo_warnx("got \"%s\", expected \"%s\"", result, expected);
	++errors;
	goto done;
    }

done:
    free(result);
    *ntests_out = ntests;
    *errors_out = errors;
}

int
main(int argc, char *argv[])
{
    int ntests = 0, errors = 0;

    initprogname(argc > 0 ? argv[0] : "check_serialize_list");

    test_serialize_list(&ntests, &errors);

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }

    exit(errors);
}
