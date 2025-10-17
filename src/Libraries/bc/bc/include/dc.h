/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#ifndef BC_DC_H
#define BC_DC_H

#if DC_ENABLED

#include <status.h>
#include <lex.h>
#include <parse.h>

/**
 * The main function for dc. It just sets variables and passes its arguments
 * through to @a bc_vm_boot().
 * @return  A status.
 */
BcStatus
dc_main(int argc, char* argv[]);

// A reference to the dc help text.
extern const char dc_help[];

/**
 * The @a BcLexNext function for dc. (See include/lex.h for a definition of
 * @a BcLexNext.)
 * @param l  The lexer.
 */
void
dc_lex_token(BcLex* l);

/**
 * Returns true if the negative char `_` should be treated as a command or not.
 * dc considers negative a command if it does *not* immediately proceed a
 * number. Otherwise, it's just considered a negative.
 * @param l  The lexer.
 * @return   True if a negative should be treated as a command, false if it
 *           should be treated as a negative sign on a number.
 */
bool
dc_lex_negCommand(BcLex* l);

// References to the signal message and its length.
extern const char dc_sig_msg[];
extern const uchar dc_sig_msg_len;

// References to an array and its length. This array is an array of lex tokens
// that, when encountered, should be treated as commands that take a register.
extern const uint8_t dc_lex_regs[];
extern const size_t dc_lex_regs_len;

// References to an array of tokens and its length. This array corresponds to
// the ASCII table, starting at double quotes. This makes it easy to look up
// tokens for characters.
extern const uint8_t dc_lex_tokens[];
extern const uint8_t dc_parse_insts[];

/**
 * The @a BcParseParse function for dc. (See include/parse.h for a definition of
 * @a BcParseParse.)
 * @param p  The parser.
 */
void
dc_parse_parse(BcParse* p);

/**
 * The @a BcParseExpr function for dc. (See include/parse.h for a definition of
 * @a BcParseExpr.)
 * @param p      The parser.
 * @param flags  Flags that define the requirements that the parsed code must
 *               meet or an error will result. See @a BcParseExpr for more info.
 */
void
dc_parse_expr(BcParse* p, uint8_t flags);

#endif // DC_ENABLED

#endif // BC_DC_H
