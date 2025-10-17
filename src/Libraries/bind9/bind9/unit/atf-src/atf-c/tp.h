/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#if !defined(ATF_C_TP_H)
#define ATF_C_TP_H

#include <stdbool.h>

#include <atf-c/error_fwd.h>

struct atf_tc;

/* ---------------------------------------------------------------------
 * The "atf_tp" type.
 * --------------------------------------------------------------------- */

struct atf_tp_impl;
struct atf_tp {
    struct atf_tp_impl *pimpl;
};
typedef struct atf_tp atf_tp_t;

/* Constructors/destructors. */
atf_error_t atf_tp_init(atf_tp_t *, const char *const *);
void atf_tp_fini(atf_tp_t *);

/* Getters. */
char **atf_tp_get_config(const atf_tp_t *);
bool atf_tp_has_tc(const atf_tp_t *, const char *);
const struct atf_tc *atf_tp_get_tc(const atf_tp_t *, const char *);
const struct atf_tc *const *atf_tp_get_tcs(const atf_tp_t *);

/* Modifiers. */
atf_error_t atf_tp_add_tc(atf_tp_t *, struct atf_tc *);

/* ---------------------------------------------------------------------
 * Free functions.
 * --------------------------------------------------------------------- */

atf_error_t atf_tp_run(const atf_tp_t *, const char *, const char *);
atf_error_t atf_tp_cleanup(const atf_tp_t *, const char *);

#endif /* ATF_C_TP_H */
