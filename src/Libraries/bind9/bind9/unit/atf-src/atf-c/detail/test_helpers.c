/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "atf-c/build.h"
#include "atf-c/check.h"
#include "atf-c/config.h"
#include "atf-c/error.h"
#include "atf-c/macros.h"

#include "dynstr.h"
#include "fs.h"
#include "process.h"
#include "test_helpers.h"

static
void
build_check_c_o_aux(const char *path, const char *failmsg,
                    const bool expect_pass)
{
    bool success;
    atf_dynstr_t iflag;
    const char *optargs[4];

    RE(atf_dynstr_init_fmt(&iflag, "-I%s", atf_config_get("atf_includedir")));

    optargs[0] = atf_dynstr_cstring(&iflag);
    optargs[1] = "-Wall";
    optargs[2] = "-Werror";
    optargs[3] = NULL;

    RE(atf_check_build_c_o(path, "test.o", optargs, &success));

    atf_dynstr_fini(&iflag);

    if ((expect_pass && !success) || (!expect_pass && success))
        atf_tc_fail("%s", failmsg);
}

void
build_check_c_o(const atf_tc_t *tc, const char *sfile, const char *failmsg,
                const bool expect_pass)
{
    atf_fs_path_t path;

    RE(atf_fs_path_init_fmt(&path, "%s/%s",
                            atf_tc_get_config_var(tc, "srcdir"), sfile));
    build_check_c_o_aux(atf_fs_path_cstring(&path), failmsg, expect_pass);
    atf_fs_path_fini(&path);
}

void
header_check(const char *hdrname)
{
    FILE *srcfile;
    char failmsg[128];

    srcfile = fopen("test.c", "w");
    ATF_REQUIRE(srcfile != NULL);
    fprintf(srcfile, "#include <%s>\n", hdrname);
    fclose(srcfile);

    snprintf(failmsg, sizeof(failmsg),
             "Header check failed; %s is not self-contained", hdrname);

    build_check_c_o_aux("test.c", failmsg, true);
}

void
get_process_helpers_path(const atf_tc_t *tc, const bool is_detail,
                         atf_fs_path_t *path)
{
    RE(atf_fs_path_init_fmt(path, "%s/%sprocess_helpers",
                            atf_tc_get_config_var(tc, "srcdir"),
                            is_detail ? "" : "detail/"));
}

struct run_h_tc_data {
    atf_tc_t *m_tc;
    const char *m_resname;
};

static
void
run_h_tc_child(void *v)
{
    struct run_h_tc_data *data = (struct run_h_tc_data *)v;

    RE(atf_tc_run(data->m_tc, data->m_resname));
}

/* TODO: Investigate if it's worth to add this functionality as part of
 * the public API.  I.e. a function to easily run a test case body in a
 * subprocess. */
void
run_h_tc(atf_tc_t *tc, const char *outname, const char *errname,
         const char *resname)
{
    atf_fs_path_t outpath, errpath;
    atf_process_stream_t outb, errb;
    atf_process_child_t child;
    atf_process_status_t status;

    RE(atf_fs_path_init_fmt(&outpath, outname));
    RE(atf_fs_path_init_fmt(&errpath, errname));

    struct run_h_tc_data data = { tc, resname };

    RE(atf_process_stream_init_redirect_path(&outb, &outpath));
    RE(atf_process_stream_init_redirect_path(&errb, &errpath));
    RE(atf_process_fork(&child, run_h_tc_child, &outb, &errb, &data));
    atf_process_stream_fini(&errb);
    atf_process_stream_fini(&outb);

    RE(atf_process_child_wait(&child, &status));
    ATF_CHECK(atf_process_status_exited(&status));
    atf_process_status_fini(&status);

    atf_fs_path_fini(&errpath);
    atf_fs_path_fini(&outpath);
}
