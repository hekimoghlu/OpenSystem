/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_extended_gcd.h"

pas_extended_gcd_result pas_extended_gcd(int64_t left, int64_t right)
{
    /* Source: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm */
    
    int64_t s, t, r;
    int64_t old_s, old_t, old_r;
    pas_extended_gcd_result result;
    
    /* We will see these common cases. */
    if (left == 1) {
        result.left_bezout_coefficient = 1;
        result.right_bezout_coefficient = 0;
        result.result = 1;
        return result;
    }
    
    if (right == 1) {
        result.left_bezout_coefficient = 0;
        result.right_bezout_coefficient = 1;
        result.result = 1;
        return result;
    }
    
    s = 0;
    old_s = 1;
    t = 1;
    old_t = 0;
    r = right;
    old_r = left;
    
    while (r) {
        int64_t quotient;
        int64_t prev_s, prev_t, prev_r;

        quotient = old_r / r;
        
        prev_r = r;
        r = old_r - quotient * r;
        old_r = prev_r;
        
        prev_s = s;
        s = old_s - quotient * s;
        old_s = prev_s;
        
        prev_t = t;
        t = old_t - quotient * t;
        old_t = prev_t;
    }
    
    result.left_bezout_coefficient = old_s;
    result.right_bezout_coefficient = old_t;
    result.result = old_r;
    return result;
}

#endif /* LIBPAS_ENABLED */
