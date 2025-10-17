/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
#if defined(TESTS_ATF_ATF_C_TEST_HELPERS_H)
#   error "Cannot include test_helpers.h more than once."
#else
#   define TESTS_ATF_ATF_C_TEST_HELPERS_H
#endif

#include <stdbool.h>

#include "atf-c/error_fwd.h"

struct atf_dynstr;
struct atf_fs_path;

#define CE(stm) ATF_CHECK(!atf_is_error(stm))
#define RE(stm) ATF_REQUIRE(!atf_is_error(stm))

#define HEADER_TC(name, hdrname) \
    ATF_TC(name); \
    ATF_TC_HEAD(name, tc) \
    { \
        atf_tc_set_md_var(tc, "descr", "Tests that the " hdrname " file can " \
            "be included on its own, without any prerequisites"); \
    } \
    ATF_TC_BODY(name, tc) \
    { \
        header_check(hdrname); \
    }

#define BUILD_TC(name, sfile, descr, failmsg) \
    ATF_TC(name); \
    ATF_TC_HEAD(name, tc) \
    { \
        atf_tc_set_md_var(tc, "descr", descr); \
    } \
    ATF_TC_BODY(name, tc) \
    { \
        build_check_c_o(tc, sfile, failmsg, true);   \
    }

#define BUILD_TC_FAIL(name, sfile, descr, failmsg) \
    ATF_TC(name); \
    ATF_TC_HEAD(name, tc) \
    { \
        atf_tc_set_md_var(tc, "descr", descr); \
    } \
    ATF_TC_BODY(name, tc) \
    { \
        build_check_c_o(tc, sfile, failmsg, false);   \
    }

void build_check_c_o(const atf_tc_t *, const char *, const char *, const bool);
void header_check(const char *);
void get_process_helpers_path(const atf_tc_t *, const bool,
                              struct atf_fs_path *);
bool read_line(int, struct atf_dynstr *);
void run_h_tc(atf_tc_t *, const char *, const char *, const char *);
