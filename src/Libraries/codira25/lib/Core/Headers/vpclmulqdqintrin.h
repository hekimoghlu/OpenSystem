/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#ifndef __IMMINTRIN_H
#error "Never use <vpclmulqdqintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __VPCLMULQDQINTRIN_H
#define __VPCLMULQDQINTRIN_H

#define _mm256_clmulepi64_epi128(A, B, I) \
  ((__m256i)__builtin_ia32_pclmulqdq256((__v4di)(__m256i)(A),  \
                                        (__v4di)(__m256i)(B),  \
                                        (char)(I)))

#ifdef __AVX512FINTRIN_H
#define _mm512_clmulepi64_epi128(A, B, I) \
  ((__m512i)__builtin_ia32_pclmulqdq512((__v8di)(__m512i)(A),  \
                                        (__v8di)(__m512i)(B),  \
                                        (char)(I)))
#endif // __AVX512FINTRIN_H

#endif /* __VPCLMULQDQINTRIN_H */

