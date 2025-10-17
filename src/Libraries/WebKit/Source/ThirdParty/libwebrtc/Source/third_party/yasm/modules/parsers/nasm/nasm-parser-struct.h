/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 18, 2023.
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
#ifndef YASM_NASM_PARSER_STRUCT_H
#define YASM_NASM_PARSER_STRUCT_H

typedef union {
    unsigned int int_info;
    char *str_val;
    yasm_intnum *intn;
    yasm_floatnum *flt;
    yasm_bytecode *bc;
    uintptr_t arch_data;
    struct {
        char *contents;
        size_t len;
    } str;
} nasm_yystype;

typedef struct yasm_parser_nasm {
    int tasm;
    int masm;

    /*@only@*/ yasm_object *object;

    /* last "base" label for local (.) labels */
    /*@null@*/ char *locallabel_base;
    size_t locallabel_base_len;

    /*@dependent@*/ yasm_preproc *preproc;
    /*@dependent@*/ yasm_errwarns *errwarns;

    /*@dependent@*/ yasm_linemap *linemap;

    /*@null@*/ yasm_bytecode *prev_bc;

    int save_input;

    yasm_scanner s;
    int state;

    int token;          /* enum tokentype or any character */
    nasm_yystype tokval;
    char tokch;         /* first character of token */

    /* one token of lookahead; used sparingly */
    int peek_token;     /* NONE if none */
    nasm_yystype peek_tokval;
    char peek_tokch;

    /* Starting point of the absolute section.  NULL if not in an absolute
     * section.
     */
    /*@null@*/ yasm_expr *absstart;

    /* Current location inside an absolute section (including the start).
     * NULL if not in an absolute section.
     */
    /*@null@*/ yasm_expr *abspos;
} yasm_parser_nasm;

#endif
