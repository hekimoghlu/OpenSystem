/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
/*
 * common.c - code common to nasm and ndisasm
 */

#include "compiler.h"
#include "nasm.h"
#include "nasmlib.h"
#include "insns.h"

/*
 * Per-pass global (across segments) state
 */
struct globalopt globl;

/*
 * Name of a register token, if applicable; otherwise NULL
 */
const char *register_name(int token)
{
    if (is_register(token))
        return nasm_reg_names[token - EXPR_REG_START];
    else
        return NULL;
}

/*
 * Common list of prefix names; ideally should be auto-generated
 * from tokens.dat. This MUST match the enum in include/nasm.h.
 */
const char *prefix_name(int token)
{
    static const char * const
        prefix_names[PREFIX_ENUM_LIMIT - PREFIX_ENUM_START] = {
        "a16", "a32", "a64", "asp", "lock", "o16", "o32", "o64", "osp",
        "rep", "repe", "repne", "repnz", "repz", "wait",
        "xacquire", "xrelease", "bnd", "nobnd", "{rex}", "{rex2}",
        "{evex}", "{vex}", "{vex3}", "{vex2}", "{nf}", "{zu}",
        "{pt}", "{pn}"
    };
    const char *name;

    /* A register can also be a prefix */
    name = register_name(token);

    if (!name) {
        const unsigned int prefix = token - PREFIX_ENUM_START;
        if (prefix < ARRAY_SIZE(prefix_names))
            name = prefix_names[prefix];
    }

    return name;
}

/*
 * True for a valid hinting-NOP opcode, after 0F.
 */
bool is_hint_nop(uint64_t op)
{
    if (op >> 16)
        return false;

    if ((op >> 8) == 0x0f)
        op = (uint8_t)op;
    else if (op >> 8)
        return false;

    return ((op & ~7) == 0x18) || (op == 0x0d);
}
