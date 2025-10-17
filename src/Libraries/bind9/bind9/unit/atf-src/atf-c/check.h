/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 26, 2024.
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
#if !defined(ATF_C_CHECK_H)
#define ATF_C_CHECK_H

#include <stdbool.h>

#include <atf-c/error_fwd.h>

/* ---------------------------------------------------------------------
 * The "atf_check_result" type.
 * --------------------------------------------------------------------- */

struct atf_check_result_impl;
struct atf_check_result {
    struct atf_check_result_impl *pimpl;
};
typedef struct atf_check_result atf_check_result_t;

/* Construtors and destructors */
void atf_check_result_fini(atf_check_result_t *);

/* Getters */
const char *atf_check_result_stdout(const atf_check_result_t *);
const char *atf_check_result_stderr(const atf_check_result_t *);
bool atf_check_result_exited(const atf_check_result_t *);
int atf_check_result_exitcode(const atf_check_result_t *);
bool atf_check_result_signaled(const atf_check_result_t *);
int atf_check_result_termsig(const atf_check_result_t *);

/* ---------------------------------------------------------------------
 * Free functions.
 * --------------------------------------------------------------------- */

atf_error_t atf_check_build_c_o(const char *, const char *,
                                const char *const [],
                                bool *);
atf_error_t atf_check_build_cpp(const char *, const char *,
                                const char *const [],
                                bool *);
atf_error_t atf_check_build_cxx_o(const char *, const char *,
                                  const char *const [],
                                  bool *);
atf_error_t atf_check_exec_array(const char *const *, atf_check_result_t *);

#endif /* ATF_C_CHECK_H */
