/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
/* @(#) st.h 5.1 89/12/14 */

#ifndef ST_INCLUDED
#define ST_INCLUDED

#if SIZEOF_VOIDP == SIZEOF_LONG
typedef unsigned long st_data_t;
#elif SIZEOF_VOIDP == SIZEOF_LONG_LONG
typedef unsigned long long st_data_t;
#endif

#define ST_DATA_T_DEFINED

typedef struct st_table st_table;

struct st_hash_type {
    int (*compare)();
    int (*hash)();
};

struct st_table {
    struct st_hash_type *type;
    int num_bins;
    int num_entries;
    struct st_table_entry **bins;
};

#define st_is_member(table,key) st_lookup(table,key,(st_data_t *)0)

enum st_retval {ST_CONTINUE, ST_STOP, ST_DELETE, ST_CHECK};

#ifndef _
# define _(args) args
#endif
#ifndef ANYARGS
# ifdef __cplusplus
#   define ANYARGS ...
# else
#   define ANYARGS
# endif
#endif

st_table *st_init_table _((struct st_hash_type *));
st_table *st_init_table_with_size _((struct st_hash_type *, int));
st_table *st_init_numtable _((void));
st_table *st_init_numtable_with_size _((int));
st_table *st_init_strtable _((void));
st_table *st_init_strtable_with_size _((int));
int st_delete _((st_table *, st_data_t *, st_data_t *));
int st_delete_safe _((st_table *, st_data_t *, st_data_t *, st_data_t));
int st_insert _((st_table *, st_data_t, st_data_t));
int st_lookup _((st_table *, st_data_t, st_data_t *));
int st_foreach _((st_table *, int (*)(ANYARGS), st_data_t));
void st_add_direct _((st_table *, st_data_t, st_data_t));
void st_free_table _((st_table *));
void st_cleanup_safe _((st_table *, st_data_t));
st_table *st_copy _((st_table *));

#define ST_NUMCMP	((int (*)()) 0)
#define ST_NUMHASH	((int (*)()) -2)

#define st_numcmp	ST_NUMCMP
#define st_numhash	ST_NUMHASH

#endif /* ST_INCLUDED */
