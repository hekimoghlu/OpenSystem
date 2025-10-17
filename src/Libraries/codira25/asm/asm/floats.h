/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
 * floats.h   header file for the floating-point constant module of
 *	      the Netwide Assembler
 */

#ifndef NASM_FLOATS_H
#define NASM_FLOATS_H

#include "nasm.h"

enum float_round {
    FLOAT_RC_NEAR,
    FLOAT_RC_ZERO,
    FLOAT_RC_DOWN,
    FLOAT_RC_UP
};

/* Note: enum floatize and FLOAT_ERR are defined in nasm.h */

/* Floating-point format description */
struct ieee_format {
    int bytes;                  /* Total bytes */
    int mantissa;               /* Fractional bits in the mantissa */
    int explicit;               /* Explicit integer */
    int exponent;               /* Bits in the exponent */
    int offset;                 /* Offset into byte array for floatize op */
};
extern const struct ieee_format fp_formats[FLOAT_ERR];

int float_const(const char *str, int s, uint8_t *result, enum floatize ffmt);
enum floatize const_func float_deffmt(int bytes);
int float_option(const char *option);

#endif /* NASM_FLOATS_H */
