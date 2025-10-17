/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#ifndef NO_WARN_X86_INTRINSICS
/* This header is distributed to simplify porting x86_64 code that
   makes explicit use of Intel intrinsics to powerpc64le.
   It is the user's responsibility to determine if the results are
   acceptable and make additional changes as necessary.
   Note that much code that uses Intel intrinsics can be rewritten in
   standard C or GNU C extensions, which are more portable and better
   optimized across multiple targets.  */
#endif

#ifndef NMMINTRIN_H_
#define NMMINTRIN_H_

/* We just include SSE4.1 header file.  */
#include <smmintrin.h>

#endif /* NMMINTRIN_H_ */
