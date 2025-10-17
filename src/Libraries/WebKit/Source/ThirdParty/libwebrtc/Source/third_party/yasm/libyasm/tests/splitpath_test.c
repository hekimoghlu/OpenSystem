/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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

#include "libyasm/file.h"

typedef struct Test_Entry {
    /* splitpath function to test */
    size_t (*splitpath) (const char *path, const char **tail);

    /* input path */
    const char *input;

    /* correct head length returned */
    size_t headlen;

    /* correct tail returned */
    const char *tail;
} Test_Entry;

static Test_Entry tests[] = {
    /* UNIX split */
    {yasm__splitpath_unix, "", 0, ""},
    {yasm__splitpath_unix, "./file.ext", 0, "file.ext"},
    {yasm__splitpath_unix, "../../file.ext", 5, "file.ext"},
    {yasm__splitpath_unix, "file.ext", 0, "file.ext"},
    {yasm__splitpath_unix, "/file.ext", 1, "file.ext"},
    {yasm__splitpath_unix, "/foo/file.ext", 4, "file.ext"},
    {yasm__splitpath_unix, "/foo/bar/file.ext", 8, "file.ext"},
    {yasm__splitpath_unix, "foo/file.ext", 3, "file.ext"},
    {yasm__splitpath_unix, "foo/bar/file.ext", 7, "file.ext"},
    {yasm__splitpath_unix, "foo/bar//file.ext", 7, "file.ext"},
    {yasm__splitpath_unix, "/", 1, ""},
    {yasm__splitpath_unix, "/foo/", 4, ""},
    {yasm__splitpath_unix, "/foo/bar/", 8, ""},
    {yasm__splitpath_unix, "foo/", 3, ""},
    {yasm__splitpath_unix, "foo/bar/", 7, ""},
    {yasm__splitpath_unix, "foo/bar//", 7, ""},
    /* Windows split */
    {yasm__splitpath_win, "", 0, ""},
    {yasm__splitpath_win, "file.ext", 0, "file.ext"},
    {yasm__splitpath_win, "./file.ext", 0, "file.ext"},
    {yasm__splitpath_win, "/file.ext", 1, "file.ext"},
    {yasm__splitpath_win, "/foo/file.ext", 4, "file.ext"},
    {yasm__splitpath_win, "/foo/bar/file.ext", 8, "file.ext"},
    {yasm__splitpath_win, "foo/file.ext", 3, "file.ext"},
    {yasm__splitpath_win, "foo/bar/file.ext", 7, "file.ext"},
    {yasm__splitpath_win, "foo/bar//file.ext", 7, "file.ext"},
    {yasm__splitpath_win, "..\\..\\file.ext", 5, "file.ext"},
    {yasm__splitpath_win, "c:file.ext", 2, "file.ext"},
    {yasm__splitpath_win, "c:.\\file.ext", 2, "file.ext"},
    {yasm__splitpath_win, "d:/file.ext", 3, "file.ext"},
    {yasm__splitpath_win, "e:/foo/file.ext", 6, "file.ext"},
    {yasm__splitpath_win, "f:/foo/bar/file.ext", 10, "file.ext"},
    {yasm__splitpath_win, "g:foo/file.ext", 5, "file.ext"},
    {yasm__splitpath_win, "h:foo/bar/file.ext", 9, "file.ext"},
    {yasm__splitpath_win, "i:foo/bar//file.ext", 9, "file.ext"},
    {yasm__splitpath_win, "d:\\file.ext", 3, "file.ext"},
    {yasm__splitpath_win, "e:\\foo/file.ext", 6, "file.ext"},
    {yasm__splitpath_win, "f:/foo\\bar\\file.ext", 10, "file.ext"},
    {yasm__splitpath_win, "g:foo\\file.ext", 5, "file.ext"},
    {yasm__splitpath_win, "h:foo/bar\\file.ext", 9, "file.ext"},
    {yasm__splitpath_win, "i:foo\\bar//\\file.ext", 9, "file.ext"},
    {yasm__splitpath_win, "\\", 1, ""},
    {yasm__splitpath_win, "c:", 2, ""},
    {yasm__splitpath_win, "d:\\", 3, ""},
    {yasm__splitpath_win, "e:\\foo/", 6, ""},
    {yasm__splitpath_win, "f:/foo\\bar\\", 10, ""},
    {yasm__splitpath_win, "g:foo\\", 5, ""},
    {yasm__splitpath_win, "h:foo/bar\\", 9, ""},
    {yasm__splitpath_win, "i:foo\\bar//\\", 9, ""},
};

static char failed[1000];
static char failmsg[100];

static int
run_test(Test_Entry *test)
{
    size_t headlen;
    const char *tail;
    const char *funcname;

    if (test->splitpath == &yasm__splitpath_unix)
        funcname = "unix";
    else
        funcname = "win";

    headlen = test->splitpath(test->input, &tail);
    if (headlen != test->headlen) {
        sprintf(failmsg,
                "splitpath_%s(\"%s\") bad head len: expected %lu, got %lu!",
                funcname, test->input, (unsigned long)test->headlen,
                (unsigned long)headlen);
        return 1;
    }

    if (strcmp(tail, test->tail) != 0) {
        sprintf(failmsg,
                "splitpath_%s(\"%s\") bad tail: expected \"%s\", got \"%s\"!",
                funcname, test->input, test->tail, tail);
        return 1;
    }

    return 0;
}

int
main(void)
{
    int nf = 0;
    int numtests = sizeof(tests)/sizeof(Test_Entry);
    int i;

    failed[0] = '\0';
    printf("Test splitpath_test: ");
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
    return (nf == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
