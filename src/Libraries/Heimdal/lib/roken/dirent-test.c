/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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
#include <stdarg.h>
#include <direct.h>
#include <errno.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include "dirent.h"

/* Note that we create a known directory structure in a subdirectory
   of the current directory to run our tests. */

#define TESTDIR "dirent-test-dir"

const char * dir_entries[] = {
    "A",
    "B",
    "C",
    "CAA",
    "CAAA",
    "CABBBB",
    "CAABBB.txt",
    "A filename with spaces"
};

const char * entries_begin_with_C[] = {
    "C",
    "CAA",
    "CAAA",
    "CABBBB",
    "CAABBB.txt"
};

const char * entries_end_with_A[] = {
    "A",
    "CAA",
    "CAAA"
};

const int n_dir_entries = sizeof(dir_entries)/sizeof(dir_entries[0]);

int teardown_test(void);

void fail_test(const char * reason, ...)
{
    va_list args;

    va_start(args, reason);
    vfprintf(stderr, reason, args);
    va_end(args);

    fprintf(stderr, " : errno = %d (%s)\n", errno, strerror(errno));
    teardown_test();
    abort();
}

void fail_test_nf(const char * format, ...)
{
    va_list args;

    fprintf(stderr, "FAIL:");

    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fprintf(stderr, " : errno = %d (%s)\n", errno, strerror(errno));
}

int touch(const char * filename)
{
    int fd;

    fd = _open(filename, _O_CREAT, _S_IREAD| _S_IWRITE);

    if (fd == -1)
        return -1;

    return _close(fd);
}

int setup_test(void)
{
    int i;

    fprintf(stderr, "Creating test directory %s ...\n", TESTDIR);

    if (_mkdir(TESTDIR))
        fail_test("Can't create test directory \"" TESTDIR "\"");

    if (_chdir(TESTDIR))
        fail_test("Can't change to test directory");

    for (i=0; i < n_dir_entries; i++) {
        if (touch(dir_entries[i]))
            fail_test("Can't create test file '%s'", dir_entries[i]);
    }

    fprintf(stderr, "Done with test setup.\n");

    return 0;
}

int teardown_test(void)
{
    char dirname[_MAX_PATH];
    size_t len;
    int i;

    printf ("Begin cleanup...\n");

    if (_getcwd(dirname, sizeof(dirname)/sizeof(char)) != NULL &&

        (len = strlen(dirname)) > sizeof(TESTDIR)/sizeof(char) &&

        !strcmp(dirname + len + 1 - sizeof(TESTDIR)/sizeof(char), TESTDIR)) {

        /* fallthrough */

    } else {
        /* did we create the directory? */

        if (!_rmdir( TESTDIR )) {
            fprintf(stderr, "Removed test directory\n");
            return 0;
        } else {
            if (errno == ENOTEMPTY) {
                if (_chdir(TESTDIR)) {
                    fprintf(stderr, "Can't change to test directory. Aborting cleanup.\n");
                    return -1;
                } else {
                    /* fallthrough */
                }
            } else {
                return -1;
            }
        }
    }

    fprintf(stderr, "Cleaning up test directory %s ...\n", TESTDIR);

    for (i=0; i < n_dir_entries; i++) {
        if (_unlink(dir_entries[i])) {
            /* if the test setup failed, we expect this to happen for
               at least some files */
        }
    }

    if (_chdir("..")) {
        fprintf(stderr, "Can't escape test directory. Giving in.\n");
        return -1;
    }

    if (_rmdir( TESTDIR )) {
        fprintf(stderr, "Can't remove test directory.\n");
        return -1;
    }

    printf("Cleaned up test directory\n");
    return 0;
}

int check_list(const char * filespec, const char ** list, int n, int expect_dot_and_dotdot)
{
    DIR * d;
    struct dirent * e;
    int n_found = 0;
    int i;
    int rv = 0;
    int retry = 1;

    d = opendir(filespec);
    if (d == NULL) {
        fail_test_nf("opendir failed for [%s]", filespec);
        return -1;
    }

    printf("Checking filespec [%s]... ", filespec);

 retry:
    while ((e = readdir(d)) != NULL) {
        n_found ++;

        if (expect_dot_and_dotdot &&
            (!strcmp(e->d_name, ".") ||
             !strcmp(e->d_name, "..")))
            continue;

        for (i=0; i < n; i++) {
            if (!strcmp(list[i], e->d_name))
                break;
        }

        if (i == n) {
            fail_test_nf("Found unexpected entry [%s]", e->d_name);
            rv = -1;
        }
    }

    if (n_found != n) {
        fail_test_nf("Unexpected number of entries [%d].  Expected %d", n_found, n);
        rv = -1;
    }

    if (retry) {
        retry = 0;
        n_found = 0;

        rewinddir(d);
        goto retry;
    }

    if (closedir(d)) {
        fail_test_nf("closedir() failed");
    }

    printf("done\n");

    return rv;
}

int run_tests()
{
    /* assumes that the test directory has been set up and we have
       changed into the test directory. */

    check_list("*", dir_entries, n_dir_entries + 2, 1);
    check_list("*.*", dir_entries, n_dir_entries + 2, 1);
    check_list("C*", entries_begin_with_C, sizeof(entries_begin_with_C)/sizeof(entries_begin_with_C[0]), 0);
    check_list("*A", entries_end_with_A, sizeof(entries_end_with_A)/sizeof(entries_end_with_A[0]), 0);

    return 0;
}

int main(int argc, char ** argv)
{
    if (setup_test())
        return 1;

    run_tests();

    teardown_test();

    return 0;
}
