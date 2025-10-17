/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "libyasm/errwarn.h"
#include "libyasm/file.h"

typedef struct Test_Entry {
    /* input string */
    const char *input;

    /* input length */
    size_t in_len;

    /* correct output string */
    const char *result;

    /* correct output length */
    size_t result_len;

    /* expected warning, if any */
    const char *warn;
} Test_Entry;

static Test_Entry tests[] = {
    {"noescape", 8, "noescape", 8, NULL},
    {"noescape2", 10, "noescape2", 10, NULL},   /* includes trailing zero */
    {"\\\\\\b\\f\\n\\r\\t\\\"", 14, "\\\b\f\n\r\t\"", 7, NULL},
    {"\\a", 2, "a", 1, NULL},
    /* hex tests */
    {"\\x", 2, "\x00", 1, NULL},
    {"\\x12", 4, "\x12", 1, NULL},
    {"\\x1234", 6, "\x34", 1, NULL},
    {"\\xg", 3, "\x00g", 2, NULL},
    {"\\xaga", 5, "\x0aga", 3, NULL},
    {"\\xaag", 5, "\xaag", 2, NULL},
    {"\\xaaa", 5, "\xaa", 1, NULL},
    {"\\x55559", 7, "\x59", 1, NULL},

    /* oct tests */
    {"\\778", 4, "\000", 1, "octal value out of range"},
    {"\\779", 4, "\001", 1, "octal value out of range"},
    {"\\1x", 3, "\001x", 2, NULL},
    {"\\7779", 5, "\xff" "9", 2, NULL},
    {"\\7999", 5, "\x11" "9", 2, "octal value out of range"},
    {"\\77a", 4, "\077a", 2, NULL},
    {"\\5555555", 8, "\x6d" "5555", 5, NULL},
    {"\\9999", 5, "\x91" "9", 2, "octal value out of range"},
};

static char failed[1000];
static char failmsg[100];

static int
run_test(Test_Entry *test)
{
    char str[256];
    size_t len;
    yasm_warn_class wclass;
    char *wstr;

    strncpy(str, test->input, test->in_len);
    len = test->in_len;

    yasm_unescape_cstring((unsigned char *)str, &len);
    if (len != test->result_len) {
        sprintf(failmsg,
                "unescape_cstring(\"%s\", %lu) bad output len: expected %lu, got %lu!",
                test->input, (unsigned long)test->in_len,
                (unsigned long)test->result_len, (unsigned long)len);
        return 1;
    }

    if (strncmp(str, test->result, len) != 0) {
        sprintf(failmsg,
                "unescape_cstring(\"%s\", %lu) bad output: expected \"%s\", got \"%s\"!",
                test->input, (unsigned long)test->in_len, test->result, str);
        return 1;
    }

    yasm_warn_fetch(&wclass, &wstr);
    if (wstr != NULL && test->warn == NULL) {
        sprintf(failmsg,
                "unescape_cstring(\"%s\", %lu) unexpected warning: %s!",
                test->input, (unsigned long)test->in_len, wstr);
        return 1;
    }
    if (wstr == NULL && test->warn != NULL) {
        sprintf(failmsg,
                "unescape_cstring(\"%s\", %lu) expected warning: %s, did not get it!",
                test->input, (unsigned long)test->in_len, test->warn);
        return 1;
    }
    if (wstr && test->warn && strcmp(wstr, test->warn) != 0) {
        sprintf(failmsg,
                "unescape_cstring(\"%s\", %lu) expected warning: %s, got %s!",
                test->input, (unsigned long)test->in_len, test->warn, wstr);
        return 1;
    }
    yasm_xfree(wstr);

    return 0;
}

int
main(void)
{
    int nf = 0;
    int numtests = sizeof(tests)/sizeof(Test_Entry);
    int i;

    yasm_errwarn_initialize();

    failed[0] = '\0';
    printf("Test uncstring_test: ");
    for (i=0; i<numtests; i++) {
        int fail = run_test(&tests[i]);
        printf("%c", fail>0 ? 'F':'.');
        fflush(stdout);
        if (fail)
            sprintf(failed, "%s ** F: %s\n", failed, failmsg);
        nf += fail;
    }

    printf(" +%d-%d/%d %d%%\n%s",
           numtests-nf, nf, numtests, 100*(numtests-nf)/numtests, failed);

    yasm_errwarn_cleanup();
    return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
