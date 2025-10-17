/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#ifndef RUBY_ID_TABLE_H
#define RUBY_ID_TABLE_H 1
#include "ruby/ruby.h"

struct rb_id_table;

/* compatible with ST_* */
enum rb_id_table_iterator_result {
    ID_TABLE_CONTINUE = ST_CONTINUE,
    ID_TABLE_STOP     = ST_STOP,
    ID_TABLE_DELETE   = ST_DELETE,
    ID_TABLE_ITERATOR_RESULT_END
};

struct rb_id_table *rb_id_table_create(size_t size);
void rb_id_table_free(struct rb_id_table *tbl);
void rb_id_table_clear(struct rb_id_table *tbl);

size_t rb_id_table_size(const struct rb_id_table *tbl);
size_t rb_id_table_memsize(const struct rb_id_table *tbl);

int rb_id_table_insert(struct rb_id_table *tbl, ID id, VALUE val);
int rb_id_table_lookup(struct rb_id_table *tbl, ID id, VALUE *valp);
int rb_id_table_delete(struct rb_id_table *tbl, ID id);

typedef enum rb_id_table_iterator_result rb_id_table_foreach_func_t(ID id, VALUE val, void *data);
typedef enum rb_id_table_iterator_result rb_id_table_foreach_values_func_t(VALUE val, void *data);
void rb_id_table_foreach(struct rb_id_table *tbl, rb_id_table_foreach_func_t *func, void *data);
void rb_id_table_foreach_values(struct rb_id_table *tbl, rb_id_table_foreach_values_func_t *func, void *data);

#endif	/* RUBY_ID_TABLE_H */
