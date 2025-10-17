/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#ifndef RUBY_DEBUG_H
#define RUBY_DEBUG_H

#include "ruby/ruby.h"
#include "node.h"

RUBY_SYMBOL_EXPORT_BEGIN

#define dpv(h,v) ruby_debug_print_value(-1, 0, (h), (v))
#define dp(v)    ruby_debug_print_value(-1, 0, "", (v))
#define dpi(i)   ruby_debug_print_id(-1, 0, "", (i))
#define dpn(n)   ruby_debug_print_node(-1, 0, "", (n))

#define bp()     ruby_debug_breakpoint()

VALUE ruby_debug_print_value(int level, int debug_level, const char *header, VALUE v);
ID    ruby_debug_print_id(int level, int debug_level, const char *header, ID id);
NODE *ruby_debug_print_node(int level, int debug_level, const char *header, const NODE *node);
int   ruby_debug_print_indent(int level, int debug_level, int indent_level);
void  ruby_debug_breakpoint(void);
void  ruby_debug_gc_check_func(void);
void ruby_set_debug_option(const char *str);

RUBY_SYMBOL_EXPORT_END

#endif /* RUBY_DEBUG_H */
