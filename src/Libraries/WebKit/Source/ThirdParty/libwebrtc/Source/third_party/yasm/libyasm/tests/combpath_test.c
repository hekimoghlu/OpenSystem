/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
#include "libyasm/coretype.h"

typedef struct Test_Entry {
    /* combpath function to test */
    char * (*combpath) (const char *from, const char *to);

    /* input "from" path */
    const char *from;

    /* input "to" path */
    const char *to;

    /* correct path returned */
    const char *out;
} Test_Entry;

static Test_Entry tests[] = {
    /* UNIX */
    {yasm__combpath_unix, "file1", "file2", "file2"},
    {yasm__combpath_unix, "./file1.ext", "./file2.ext", "file2.ext"},
    {yasm__combpath_unix, "/file1", "file2", "/file2"},
    {yasm__combpath_unix, "file1", "/file2", "/file2"},
    {yasm__combpath_unix, "/foo/file1", "../../file2", "/file2"},
    {yasm__combpath_unix, "/foo//file1", "../../file2", "/file2"},
    {yasm__combpath_unix, "foo/bar/file1", "../file2", "foo/file2"},
    {yasm__combpath_unix, "foo/bar/file1", "../../../file2", "../file2"},
    {yasm__combpath_unix, "foo/bar//file1", "../..//..//file2", "../file2"},
    {yasm__combpath_unix, "foo/bar/", "file2", "foo/bar/file2"},
    {yasm__combpath_unix, "../../file1", "../../file2", "../../../../file2"},
    {yasm__combpath_unix, "../foo/bar/../file1", "../../file2", "../foo/bar/../../../file2"},
    {yasm__combpath_unix, "/", "../file2", "/file2"},
    {yasm__combpath_unix, "../foo/", "../file2", "../file2"},
    {yasm__combpath_unix, "../foo/file1", "../../bar/file2", "../../bar/file2"},

    /* Windows */
    {yasm__combpath_win, "file1", "file2", "file2"},
    {yasm__combpath_win, "./file1.ext", "./file2.ext", "file2.ext"},
    {yasm__combpath_win, "./file1.ext", ".\\file2.ext", "file2.ext"},
    {yasm__combpath_win, ".\\file1.ext", "./file2.ext", "file2.ext"},
    {yasm__combpath_win, "/file1", "file2", "\\file2"},
    {yasm__combpath_win, "\\file1", "file2", "\\file2"},
    {yasm__combpath_win, "file1", "/file2", "\\file2"},
    {yasm__combpath_win, "file1", "\\file2", "\\file2"},
    {yasm__combpath_win, "/foo\\file1", "../../file2", "\\file2"},
    {yasm__combpath_win, "\\foo\\\\file1", "..\\../file2", "\\file2"},
    {yasm__combpath_win, "foo/bar/file1", "../file2", "foo\\file2"},
    {yasm__combpath_win, "foo/bar/file1", "../..\\../file2", "..\\file2"},
    {yasm__combpath_win, "foo/bar//file1", "../..\\\\..//file2", "..\\file2"},
    {yasm__combpath_win, "foo/bar/", "file2", "foo\\bar\\file2"},
    {yasm__combpath_win, "..\\../file1", "../..\\file2", "..\\..\\..\\..\\file2"},
    {yasm__combpath_win, "../foo/bar\\\\../file1", "../..\\file2", "..\\foo\\bar\\..\\..\\..\\file2"},
    {yasm__combpath_win, "/", "../file2", "\\file2"},
    {yasm__combpath_win, "../foo/", "../file2", "..\\file2"},
    {yasm__combpath_win, "../foo/file1", "../..\\bar\\file2", "..\\..\\bar\\file2"},
    {yasm__combpath_win, "c:/file1.ext", "./file2.ext", "c:\\file2.ext"},
    {yasm__combpath_win, "e:\\path\\to/file1.ext", ".\\file2.ext", "e:\\path\\to\\file2.ext"},
    {yasm__combpath_win, ".\\file1.ext", "g:file2.ext", "g:file2.ext"},
};

static char failed[1000];
static char failmsg[100];

static int
run_test(Test_Entry *test)
{
    char *out;
    const char *funcname;

    if (test->combpath == &yasm__combpath_unix)
        funcname = "unix";
    else
        funcname = "win";

    out = test->combpath(test->from, test->to);

    if (strcmp(out, test->out) != 0) {
        sprintf(failmsg,
                "combpath_%s(\"%s\", \"%s\"): expected \"%s\", got \"%s\"!",
                funcname, test->from, test->to, test->out, out);
        yasm_xfree(out);
        return 1;
    }

    yasm_xfree(out);
    return 0;
}

int
main(void)
{
    int nf = 0;
    int numtests = sizeof(tests)/sizeof(Test_Entry);
    int i;

    failed[0] = '\0';
    printf("Test combpath_test: ");
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
