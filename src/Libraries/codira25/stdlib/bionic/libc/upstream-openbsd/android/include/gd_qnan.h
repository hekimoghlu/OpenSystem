/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
//
// The values in this file came from reading the bits of <math.h>'s NAN back from a union.
//

#define f_QNAN 0x7fc00000

#define d_QNAN0 0x00000000
#define d_QNAN1 0x7ff80000

#if defined(__LP64__)
#define ld_QNAN0 0x00000000
#define ld_QNAN1 0x00000000
#define ld_QNAN2 0x00000000
#define ld_QNAN3 0x7fff8000
#else
// LP32 sizeof(long double) == sizeof(double), so LP32 shouldn't try to use these constants.
#endif
