/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#include "ruby/ruby.h"
#include "ruby/encoding.h"

/* loadpath.c */
const char ruby_exec_prefix[] = "";
const char ruby_initial_load_paths[] = "";

/* localeinit.c */
VALUE
rb_locale_charmap(VALUE klass)
{
    /* never used */
    return Qnil;
}

int
rb_locale_charmap_index(void)
{
    return -1;
}

int
Init_enc_set_filesystem_encoding(void)
{
    return rb_enc_to_index(rb_default_external_encoding());
}

void rb_encdb_declare(const char *name);
int rb_encdb_alias(const char *alias, const char *orig);
void
Init_enc(void)
{
    rb_encdb_declare("ASCII-8BIT");
    rb_encdb_declare("US-ASCII");
    rb_encdb_declare("UTF-8");
    rb_encdb_alias("BINARY", "ASCII-8BIT");
    rb_encdb_alias("ASCII", "US-ASCII");
}
