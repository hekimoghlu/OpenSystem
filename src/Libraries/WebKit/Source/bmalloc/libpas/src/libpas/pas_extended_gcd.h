/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#ifndef PAS_EXTENDED_GCD_H
#define PAS_EXTENDED_GCD_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_extended_gcd_result;
typedef struct pas_extended_gcd_result pas_extended_gcd_result;

struct pas_extended_gcd_result {
    int64_t left_bezout_coefficient;
    int64_t right_bezout_coefficient;
    int64_t result;
};

PAS_API pas_extended_gcd_result pas_extended_gcd(int64_t left, int64_t right);

PAS_END_EXTERN_C;

#endif /* PAS_EXTENDED_GCD_H */

