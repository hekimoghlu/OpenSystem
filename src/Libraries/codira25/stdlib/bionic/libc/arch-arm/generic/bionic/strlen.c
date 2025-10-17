/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#include <string.h>
#include <stdint.h>

size_t strlen_generic(const char *s)
{
    __builtin_prefetch(s);
    __builtin_prefetch(s+32);

    union {
        const char      *b;
        const uint32_t  *w;
        uintptr_t       i;
    } u;

    // these are some scratch variables for the asm code below
    uint32_t v, t;

    // initialize the string length to zero
    size_t l = 0;

    // align the pointer to a 32-bit word boundary
    u.b = s;
    while (u.i & 0x3)  {
        if (__builtin_expect(*u.b++ == 0, 0)) {
            goto done;
        }
        l++;
    }

    // loop for each word, testing if it contains a zero byte
    // if so, exit the loop and update the length.
    // We need to process 32 bytes per loop to schedule PLD properly
    // and achieve the maximum bus speed.
    asm(
        "ldr     %[v], [%[s]], #4           \n"
        "sub     %[l], %[l], %[s]           \n"
        "0:                                 \n"
        "pld     [%[s], #64]                \n"
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
#if !defined(__OPTIMIZE_SIZE__)
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
        "sub     %[t], %[v], %[mask], lsr #7\n"
        "and     %[t], %[t], %[mask]        \n"
        "bics    %[t], %[t], %[v]           \n"
        "bne     1f                         \n"
        "ldr     %[v], [%[s]], #4           \n"
#endif
        "b       0b                         \n"
        "1:                                 \n"
        "add     %[l], %[l], %[s]           \n"
        "tst     %[v], #0xFF                \n"
        "beq     2f                         \n"
        "add     %[l], %[l], #1             \n"
        "tst     %[v], #0xFF00              \n"
        "beq     2f                         \n"
        "add     %[l], %[l], #1             \n"
        "tst     %[v], #0xFF0000            \n"
        "beq     2f                         \n"
        "add     %[l], %[l], #1             \n"
        "2:                                 \n"
        : [l]"=&r"(l), [v]"=&r"(v), [t]"=&r"(t), [s]"=&r"(u.b)
        : "%[l]"(l), "%[s]"(u.b), [mask]"r"(0x80808080UL)
        : "cc"
    );

done:
    return l;
}
