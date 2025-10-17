/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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

#ifndef RUBY_RUBY_BACKWARD_H
#define RUBY_RUBY_BACKWARD_H 1

#define RClass RClassDeprecated
#ifndef __cplusplus
DEPRECATED_TYPE(("RClass is internal use only"),
struct RClass {
    struct RBasic basic;
});
#endif

#define DECLARE_DEPRECATED_FEATURE(ver, func) \
    NORETURN(ERRORFUNC(("deprecated since "#ver), DEPRECATED(void func(void))))

/* eval.c */
DECLARE_DEPRECATED_FEATURE(2.2, rb_disable_super);
DECLARE_DEPRECATED_FEATURE(2.2, rb_enable_super);

/* hash.c */
DECLARE_DEPRECATED_FEATURE(2.2, rb_hash_iter_lev);
DECLARE_DEPRECATED_FEATURE(2.2, rb_hash_ifnone);

/* string.c */
DECLARE_DEPRECATED_FEATURE(2.2, rb_str_associate);
DECLARE_DEPRECATED_FEATURE(2.2, rb_str_associated);

/* variable.c */
DEPRECATED(void rb_autoload(VALUE, ID, const char*));

/* vm.c */
DECLARE_DEPRECATED_FEATURE(2.2, rb_clear_cache);
DECLARE_DEPRECATED_FEATURE(2.2, rb_frame_pop);

#define DECLARE_DEPRECATED_INTERNAL_FEATURE(func) \
    NORETURN(ERRORFUNC(("deprecated internal function"), DEPRECATED(void func(void))))

/* eval.c */
NORETURN(ERRORFUNC(("internal function"), void rb_frozen_class_p(VALUE)));

/* error.c */
DECLARE_DEPRECATED_INTERNAL_FEATURE(rb_compile_error);
DECLARE_DEPRECATED_INTERNAL_FEATURE(rb_compile_error_with_enc);
DECLARE_DEPRECATED_INTERNAL_FEATURE(rb_compile_error_append);

/* struct.c */
DECLARE_DEPRECATED_INTERNAL_FEATURE(rb_struct_ptr);

/* variable.c */
DECLARE_DEPRECATED_INTERNAL_FEATURE(rb_generic_ivar_table);
NORETURN(ERRORFUNC(("internal function"), VALUE rb_mod_const_missing(VALUE, VALUE)));

/* from version.c */
#ifndef RUBY_SHOW_COPYRIGHT_TO_DIE
# define RUBY_SHOW_COPYRIGHT_TO_DIE 1
#endif
#if RUBY_SHOW_COPYRIGHT_TO_DIE
/* for source code backward compatibility */
DEPRECATED(static inline int ruby_show_copyright_to_die(int));
static inline int
ruby_show_copyright_to_die(int exitcode)
{
    ruby_show_copyright();
    return exitcode;
}
#define ruby_show_copyright() /* defer EXIT_SUCCESS */ \
    (exit(ruby_show_copyright_to_die(EXIT_SUCCESS)))
#endif

#endif /* RUBY_RUBY_BACKWARD_H */
