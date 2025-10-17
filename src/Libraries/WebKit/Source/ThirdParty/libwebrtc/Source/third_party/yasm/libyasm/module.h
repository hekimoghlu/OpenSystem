/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
#ifndef YASM_MODULE_H
#define YASM_MODULE_H

#ifndef YASM_LIB_DECL
#define YASM_LIB_DECL
#endif

typedef enum yasm_module_type {
    YASM_MODULE_ARCH = 0,
    YASM_MODULE_DBGFMT,
    YASM_MODULE_OBJFMT,
    YASM_MODULE_LISTFMT,
    YASM_MODULE_PARSER,
    YASM_MODULE_PREPROC
} yasm_module_type;

YASM_LIB_DECL
/*@dependent@*/ /*@null@*/ void *yasm_load_module
    (yasm_module_type type, const char *keyword);

#define yasm_load_arch(keyword) \
    yasm_load_module(YASM_MODULE_ARCH, keyword)
#define yasm_load_dbgfmt(keyword)       \
    yasm_load_module(YASM_MODULE_DBGFMT, keyword)
#define yasm_load_objfmt(keyword)       \
    yasm_load_module(YASM_MODULE_OBJFMT, keyword)
#define yasm_load_listfmt(keyword)      \
    yasm_load_module(YASM_MODULE_LISTFMT, keyword)
#define yasm_load_parser(keyword)       \
    yasm_load_module(YASM_MODULE_PARSER, keyword)
#define yasm_load_preproc(keyword)      \
    yasm_load_module(YASM_MODULE_PREPROC, keyword)

YASM_LIB_DECL
void yasm_list_modules
    (yasm_module_type type,
     void (*printfunc) (const char *name, const char *keyword));

#define yasm_list_arch(func)            \
    yasm_list_modules(YASM_MODULE_ARCH, func)
#define yasm_list_dbgfmt(func)          \
    yasm_list_modules(YASM_MODULE_DBGFMT, func)
#define yasm_list_objfmt(func)          \
    yasm_list_modules(YASM_MODULE_OBJFMT, func)
#define yasm_list_listfmt(func)         \
    yasm_list_modules(YASM_MODULE_LISTFMT, func)
#define yasm_list_parser(func)          \
    yasm_list_modules(YASM_MODULE_PARSER, func)
#define yasm_list_preproc(func)         \
    yasm_list_modules(YASM_MODULE_PREPROC, func)

YASM_LIB_DECL
void yasm_register_module(yasm_module_type type, const char *keyword,
                          void *data);

#endif
