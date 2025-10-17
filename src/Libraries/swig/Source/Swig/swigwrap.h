/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
/* $Id: swig.h 9635 2007-01-12 01:44:16Z beazley $ */

typedef struct Wrapper {
    Hash *localh;
    String *def;
    String *locals;
    String *code;
} Wrapper;

extern Wrapper *NewWrapper(void);
extern void     DelWrapper(Wrapper *w);
extern void     Wrapper_compact_print_mode_set(int flag);
extern void     Wrapper_pretty_print(String *str, File *f);
extern void     Wrapper_compact_print(String *str, File *f);
extern void     Wrapper_print(Wrapper *w, File *f);
extern int      Wrapper_add_local(Wrapper *w, const_String_or_char_ptr name, const_String_or_char_ptr decl);
extern int      Wrapper_add_localv(Wrapper *w, const_String_or_char_ptr name, ...);
extern int      Wrapper_check_local(Wrapper *w, const_String_or_char_ptr name);
extern char    *Wrapper_new_local(Wrapper *w, const_String_or_char_ptr name, const_String_or_char_ptr decl);
extern char    *Wrapper_new_localv(Wrapper *w, const_String_or_char_ptr name, ...);
