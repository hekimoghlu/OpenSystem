/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
/* $Id: preprocessor.h 11080 2009-01-24 13:15:51Z bhy $ */

#ifndef SWIG_PREPROCESSOR_H_
#define SWIG_PREPROCESSOR_H_

#include "swigwarn.h"

#ifdef __cplusplus
extern "C" {
#endif
  extern int Preprocessor_expr(String *s, int *error);
  extern char *Preprocessor_expr_error(void);
  extern Hash *Preprocessor_define(const_String_or_char_ptr str, int swigmacro);
  extern void Preprocessor_undef(const_String_or_char_ptr name);
  extern void Preprocessor_init(void);
  extern void Preprocessor_delete(void);
  extern String *Preprocessor_parse(String *s);
  extern void Preprocessor_include_all(int);
  extern void Preprocessor_import_all(int);
  extern void Preprocessor_ignore_missing(int);
  extern void Preprocessor_error_as_warning(int);
  extern List *Preprocessor_depend(void);
  extern void Preprocessor_expr_init(void);
  extern void Preprocessor_expr_delete(void);

#ifdef __cplusplus
}
#endif
#endif
