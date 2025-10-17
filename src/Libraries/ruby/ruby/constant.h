/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#ifndef CONSTANT_H
#define CONSTANT_H

typedef enum {
    CONST_DEPRECATED = 0x100,

    CONST_VISIBILITY_MASK = 0xff,
    CONST_PUBLIC    = 0x00,
    CONST_PRIVATE,
    CONST_VISIBILITY_MAX
} rb_const_flag_t;

#define RB_CONST_PRIVATE_P(ce) \
    (((ce)->flag & CONST_VISIBILITY_MASK) == CONST_PRIVATE)
#define RB_CONST_PUBLIC_P(ce) \
    (((ce)->flag & CONST_VISIBILITY_MASK) == CONST_PUBLIC)

#define RB_CONST_DEPRECATED_P(ce) \
    ((ce)->flag & CONST_DEPRECATED)

typedef struct rb_const_entry_struct {
    rb_const_flag_t flag;
    int line;
    const VALUE value;            /* should be mark */
    const VALUE file;             /* should be mark */
} rb_const_entry_t;

VALUE rb_mod_private_constant(int argc, const VALUE *argv, VALUE obj);
VALUE rb_mod_public_constant(int argc, const VALUE *argv, VALUE obj);
VALUE rb_mod_deprecate_constant(int argc, const VALUE *argv, VALUE obj);
void rb_free_const_table(struct rb_id_table *tbl);
VALUE rb_public_const_get(VALUE klass, ID id);
VALUE rb_public_const_get_at(VALUE klass, ID id);
VALUE rb_public_const_get_from(VALUE klass, ID id);
int rb_public_const_defined(VALUE klass, ID id);
int rb_public_const_defined_at(VALUE klass, ID id);
int rb_public_const_defined_from(VALUE klass, ID id);
rb_const_entry_t *rb_const_lookup(VALUE klass, ID id);
int rb_autoloading_value(VALUE mod, ID id, VALUE *value, rb_const_flag_t *flag);

#endif /* CONSTANT_H */
