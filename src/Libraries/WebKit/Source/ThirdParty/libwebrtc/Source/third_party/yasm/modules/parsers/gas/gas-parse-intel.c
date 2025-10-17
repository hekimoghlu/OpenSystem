/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
#include <util.h>

#include <libyasm.h>

#include "modules/parsers/gas/gas-parser.h"
#include "modules/parsers/nasm/nasm-parser-struct.h"

extern yasm_bytecode *gas_intel_syntax_parse_instr(yasm_parser_nasm *parser_nasm, unsigned char *instr);

#define SET_FIELDS(to, from) \
    (to)->object = (from)->object; \
    (to)->locallabel_base = (from)->locallabel_base; \
    (to)->locallabel_base_len = (from)->locallabel_base_len; \
    (to)->preproc = (from)->preproc; \
    (to)->errwarns = (from)->errwarns; \
    (to)->linemap = (from)->linemap; \
    (to)->prev_bc = (from)->prev_bc;

yasm_bytecode *parse_instr_intel(yasm_parser_gas *parser_gas)
{
    char *stok, *slim;
    unsigned char *line;
    size_t length;

    yasm_parser_nasm parser_nasm;
    yasm_bytecode *bc;

    memset(&parser_nasm, 0, sizeof(parser_nasm));

    yasm_arch_set_var(parser_gas->object->arch, "gas_intel_mode", 1);
    SET_FIELDS(&parser_nasm, parser_gas);
    parser_nasm.masm = 1;

    stok = (char *) parser_gas->s.tok;
    slim = (char *) parser_gas->s.lim;
    length = 0;
    while (&stok[length] < slim && stok[length] != '\n') {
        length++;
    }

    if (&stok[length] == slim && parser_gas->line) {
        line = yasm_xmalloc(length + parser_gas->lineleft + 1);
        memcpy(line, parser_gas->s.tok, length);
        memcpy(line + length, parser_gas->linepos, parser_gas->lineleft);
        length += parser_gas->lineleft;
        if (line[length - 1] == '\n') length--;
    } else {
        line = yasm_xmalloc(length + 1);
        memcpy(line, parser_gas->s.tok, length);
    }
    line[length] = '\0';

    bc = gas_intel_syntax_parse_instr(&parser_nasm, line);

    SET_FIELDS(parser_gas, &parser_nasm);
    yasm_arch_set_var(parser_gas->object->arch, "gas_intel_mode", 0);

    yasm_xfree(line);

    return bc;
}
