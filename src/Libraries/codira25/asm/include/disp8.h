/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
 * disp8.h   header file for disp8.c
 */

#ifndef NASM_DISP8_H
#define NASM_DISP8_H

#include "nasm.h"

/*
 * Find shift value for compressed displacement (disp8 << shift)
 */
static inline unsigned int get_disp8_shift(const insn *ins)
{
    bool evex_b;
    unsigned int  evex_w;
    unsigned int  vectlen;
    enum ttypes   tuple = ins->evex_tuple;

    if (likely(!tuple))
        return 0;

    evex_b  = !!(ins->evex & EVEX_B);
    evex_w  = !!(ins->evex & EVEX_W);
    /* XXX: consider RC/SAE here?! */
    vectlen = getfield(EVEX_LL, ins->evex);

    switch (tuple) {
        /* Full, half vector unless broadcast */
    case FV:
        return evex_b ? 2 + evex_w : vectlen + 4;
    case HV:
        return evex_b ? 2 + evex_w : vectlen + 3;

        /* Full vector length */
    case FVM:
        return vectlen + 4;

        /* Fixed tuple lengths */
    case T1S8:
        return 0;
    case T1S16:
        return 1;
    case T1F32:
        return 2;
    case T1F64:
        return 3;
    case M128:
        return 4;

        /* One scalar */
    case T1S:
        return 2 + evex_w;

        /* 2, 4, 8 32/64-bit elements */
    case T2:
        return 3 + evex_w;
    case T4:
        return 4 + evex_w;
    case T8:
        return 5 + evex_w;

        /* Half, quarter, eigth mem */
    case HVM:
        return vectlen + 3;
    case QVM:
        return vectlen + 2;
    case OVM:
        return vectlen + 1;

        /* MOVDDUP */
    case DUP:
        return vectlen + 3;

    default:
        return 0;
    }
}

#endif  /* NASM_DISP8_H */
