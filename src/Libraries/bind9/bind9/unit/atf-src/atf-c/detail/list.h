/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#if !defined(ATF_C_LIST_H)
#define ATF_C_LIST_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>

#include <atf-c/error_fwd.h>

/* ---------------------------------------------------------------------
 * The "atf_list_citer" type.
 * --------------------------------------------------------------------- */

struct atf_list_citer {
    const struct atf_list *m_list;
    const void *m_entry;
};
typedef struct atf_list_citer atf_list_citer_t;

/* Getters. */
const void *atf_list_citer_data(const atf_list_citer_t);
atf_list_citer_t atf_list_citer_next(const atf_list_citer_t);

/* Operators. */
bool atf_equal_list_citer_list_citer(const atf_list_citer_t,
                                     const atf_list_citer_t);

/* ---------------------------------------------------------------------
 * The "atf_list_iter" type.
 * --------------------------------------------------------------------- */

struct atf_list_iter {
    struct atf_list *m_list;
    void *m_entry;
};
typedef struct atf_list_iter atf_list_iter_t;

/* Getters. */
void *atf_list_iter_data(const atf_list_iter_t);
atf_list_iter_t atf_list_iter_next(const atf_list_iter_t);

/* Operators. */
bool atf_equal_list_iter_list_iter(const atf_list_iter_t,
                                   const atf_list_iter_t);

/* ---------------------------------------------------------------------
 * The "atf_list" type.
 * --------------------------------------------------------------------- */

struct atf_list {
    void *m_begin;
    void *m_end;

    size_t m_size;
};
typedef struct atf_list atf_list_t;

/* Constructors and destructors */
atf_error_t atf_list_init(atf_list_t *);
void atf_list_fini(atf_list_t *);

/* Getters. */
atf_list_iter_t atf_list_begin(atf_list_t *);
atf_list_citer_t atf_list_begin_c(const atf_list_t *);
atf_list_iter_t atf_list_end(atf_list_t *);
atf_list_citer_t atf_list_end_c(const atf_list_t *);
void *atf_list_index(atf_list_t *, const size_t);
const void *atf_list_index_c(const atf_list_t *, const size_t);
size_t atf_list_size(const atf_list_t *);
char **atf_list_to_charpp(const atf_list_t *);

/* Modifiers. */
atf_error_t atf_list_append(atf_list_t *, void *, bool);
void atf_list_append_list(atf_list_t *, atf_list_t *);

/* Macros. */
#define atf_list_for_each(iter, list) \
    for (iter = atf_list_begin(list); \
         !atf_equal_list_iter_list_iter((iter), atf_list_end(list)); \
         iter = atf_list_iter_next(iter))
#define atf_list_for_each_c(iter, list) \
    for (iter = atf_list_begin_c(list); \
         !atf_equal_list_citer_list_citer((iter), atf_list_end_c(list)); \
         iter = atf_list_citer_next(iter))

#endif /* ATF_C_LIST_H */
