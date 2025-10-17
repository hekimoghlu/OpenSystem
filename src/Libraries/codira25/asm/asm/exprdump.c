/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
 * exprdump.c
 *
 * Debugging code to dump the contents of an expression vector to stdout
 */

#include "nasm.h"

static const char *expr_type(int32_t type)
{
    static char seg_str[64];

    switch (type) {
    case 0:
        return "null";
    case EXPR_UNKNOWN:
        return "unknown";
    case EXPR_SIMPLE:
        return "simple";
    case EXPR_WRT:
        return "wrt";
    case EXPR_RDSAE:
        return "sae";
    default:
        break;
    }

    if (type >= EXPR_REG_START && type <= EXPR_REG_END) {
        return nasm_reg_names[type - EXPR_REG_START];
    } else if (type >= EXPR_SEGBASE) {
        snprintf(seg_str, sizeof seg_str, "%sseg %d",
                 (type - EXPR_SEGBASE) == location.segment ? "this " : "",
                 type - EXPR_SEGBASE);
        return seg_str;
    } else {
        return "ERR";
    }
}

void dump_expr(const expr *e)
{
    printf("[");
    for (; e->type; e++)
        printf("<%s(%d),%"PRId64">", expr_type(e->type), e->type, e->value);
    printf("]\n");
}
