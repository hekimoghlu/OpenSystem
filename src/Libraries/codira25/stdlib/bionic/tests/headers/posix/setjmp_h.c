/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <setjmp.h>

#include "header_checks.h"

// POSIX says it's undefined whether `setjmp` is a macro or a function,
// but C11 says it's a macro, and the C standard always wins.
#if !defined(setjmp)
#error setjmp
#endif

static void setjmp_h() {
  TYPE(jmp_buf);
  TYPE(sigjmp_buf);

  FUNCTION(_longjmp, void (*f)(jmp_buf, int));
  FUNCTION(longjmp, void (*f)(jmp_buf, int));
  FUNCTION(siglongjmp, void (*f)(sigjmp_buf, int));

  FUNCTION(_setjmp, int (*f)(jmp_buf));
  FUNCTION(setjmp, int (*f)(jmp_buf));
#if defined(__GLIBC__)
  FUNCTION(__sigsetjmp, int (*f)(sigjmp_buf, int));
#else
  FUNCTION(sigsetjmp, int (*f)(sigjmp_buf, int));
#endif
}
