/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#if !defined(ATF_C_TC_H)
#define ATF_C_TC_H

#include <stdbool.h>
#include <stddef.h>

#include <atf-c/defs.h>
#include <atf-c/error_fwd.h>

struct atf_tc;

typedef void (*atf_tc_head_t)(struct atf_tc *);
typedef void (*atf_tc_body_t)(const struct atf_tc *);
typedef void (*atf_tc_cleanup_t)(const struct atf_tc *);

/* ---------------------------------------------------------------------
 * The "atf_tc_pack" type.
 * --------------------------------------------------------------------- */

/* For static initialization only. */
struct atf_tc_pack {
    const char *m_ident;

    const char *const *m_config;

    atf_tc_head_t m_head;
    atf_tc_body_t m_body;
    atf_tc_cleanup_t m_cleanup;
};
typedef const struct atf_tc_pack atf_tc_pack_t;

/* ---------------------------------------------------------------------
 * The "atf_tc" type.
 * --------------------------------------------------------------------- */

struct atf_tc_impl;
struct atf_tc {
    struct atf_tc_impl *pimpl;
};
typedef struct atf_tc atf_tc_t;

/* Constructors/destructors. */
atf_error_t atf_tc_init(atf_tc_t *, const char *, atf_tc_head_t,
                        atf_tc_body_t, atf_tc_cleanup_t,
                        const char *const *);
atf_error_t atf_tc_init_pack(atf_tc_t *, atf_tc_pack_t *,
                             const char *const *);
void atf_tc_fini(atf_tc_t *);

/* Getters. */
const char *atf_tc_get_ident(const atf_tc_t *);
const char *atf_tc_get_config_var(const atf_tc_t *, const char *);
const char *atf_tc_get_config_var_wd(const atf_tc_t *, const char *,
                                     const char *);
bool atf_tc_get_config_var_as_bool(const atf_tc_t *, const char *);
bool atf_tc_get_config_var_as_bool_wd(const atf_tc_t *, const char *,
                                      const bool);
long atf_tc_get_config_var_as_long(const atf_tc_t *, const char *);
long atf_tc_get_config_var_as_long_wd(const atf_tc_t *, const char *,
                                      const long);
const char *atf_tc_get_md_var(const atf_tc_t *, const char *);
char **atf_tc_get_md_vars(const atf_tc_t *);
bool atf_tc_has_config_var(const atf_tc_t *, const char *);
bool atf_tc_has_md_var(const atf_tc_t *, const char *);

/* Modifiers. */
atf_error_t atf_tc_set_md_var(atf_tc_t *, const char *, const char *, ...);

/* ---------------------------------------------------------------------
 * Free functions.
 * --------------------------------------------------------------------- */

atf_error_t atf_tc_run(const atf_tc_t *, const char *);
atf_error_t atf_tc_cleanup(const atf_tc_t *);

/* To be run from test case bodies only. */
void atf_tc_fail(const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 2)
    ATF_DEFS_ATTRIBUTE_NORETURN;
void atf_tc_fail_nonfatal(const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 2);
void atf_tc_pass(void)
    ATF_DEFS_ATTRIBUTE_NORETURN;
void atf_tc_require_prog(const char *);
void atf_tc_skip(const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 2)
    ATF_DEFS_ATTRIBUTE_NORETURN;
void atf_tc_expect_pass(void);
void atf_tc_expect_fail(const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 2);
void atf_tc_expect_exit(const int, const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(2, 3);
void atf_tc_expect_signal(const int, const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(2, 3);
void atf_tc_expect_death(const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 2);
void atf_tc_expect_timeout(const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(1, 2);

/* To be run from test case bodies only; internal to macros.h. */
void atf_tc_fail_check(const char *, const size_t, const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(3, 4);
void atf_tc_fail_requirement(const char *, const size_t, const char *, ...)
    ATF_DEFS_ATTRIBUTE_FORMAT_PRINTF(3, 4)
    ATF_DEFS_ATTRIBUTE_NORETURN;
void atf_tc_check_errno(const char *, const size_t, const int,
                        const char *, const bool);
void atf_tc_require_errno(const char *, const size_t, const int,
                          const char *, const bool);

#endif /* ATF_C_TC_H */
