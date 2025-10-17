/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
#if !defined(ATF_C_MAP_H)
#define ATF_C_MAP_H

#include <stdarg.h>
#include <stdbool.h>

#include <atf-c/error_fwd.h>

#include "list.h"

/* ---------------------------------------------------------------------
 * The "atf_map_citer" type.
 * --------------------------------------------------------------------- */

struct atf_map_citer {
    const struct atf_map *m_map;
    const void *m_entry;
    atf_list_citer_t m_listiter;
};
typedef struct atf_map_citer atf_map_citer_t;

/* Getters. */
const char *atf_map_citer_key(const atf_map_citer_t);
const void *atf_map_citer_data(const atf_map_citer_t);
atf_map_citer_t atf_map_citer_next(const atf_map_citer_t);

/* Operators. */
bool atf_equal_map_citer_map_citer(const atf_map_citer_t,
                                   const atf_map_citer_t);

/* ---------------------------------------------------------------------
 * The "atf_map_iter" type.
 * --------------------------------------------------------------------- */

struct atf_map_iter {
    struct atf_map *m_map;
    void *m_entry;
    atf_list_iter_t m_listiter;
};
typedef struct atf_map_iter atf_map_iter_t;

/* Getters. */
const char *atf_map_iter_key(const atf_map_iter_t);
void *atf_map_iter_data(const atf_map_iter_t);
atf_map_iter_t atf_map_iter_next(const atf_map_iter_t);

/* Operators. */
bool atf_equal_map_iter_map_iter(const atf_map_iter_t,
                                 const atf_map_iter_t);

/* ---------------------------------------------------------------------
 * The "atf_map" type.
 * --------------------------------------------------------------------- */

/* A list-based map.  Typically very inefficient, but our maps are small
 * enough. */
struct atf_map {
    atf_list_t m_list;
};
typedef struct atf_map atf_map_t;

/* Constructors and destructors */
atf_error_t atf_map_init(atf_map_t *);
atf_error_t atf_map_init_charpp(atf_map_t *, const char *const *);
void atf_map_fini(atf_map_t *);

/* Getters. */
atf_map_iter_t atf_map_begin(atf_map_t *);
atf_map_citer_t atf_map_begin_c(const atf_map_t *);
atf_map_iter_t atf_map_end(atf_map_t *);
atf_map_citer_t atf_map_end_c(const atf_map_t *);
atf_map_iter_t atf_map_find(atf_map_t *, const char *);
atf_map_citer_t atf_map_find_c(const atf_map_t *, const char *);
size_t atf_map_size(const atf_map_t *);
char **atf_map_to_charpp(const atf_map_t *);

/* Modifiers. */
atf_error_t atf_map_insert(atf_map_t *, const char *, void *, bool);

/* Macros. */
#define atf_map_for_each(iter, map) \
    for (iter = atf_map_begin(map); \
         !atf_equal_map_iter_map_iter((iter), atf_map_end(map)); \
         iter = atf_map_iter_next(iter))
#define atf_map_for_each_c(iter, map) \
    for (iter = atf_map_begin_c(map); \
         !atf_equal_map_citer_map_citer((iter), atf_map_end_c(map)); \
         iter = atf_map_citer_next(iter))

#endif /* ATF_C_MAP_H */
