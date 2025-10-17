/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_WORKAROUNDS_H
#define SUPPORT_TEST_WORKAROUNDS_H

#include "test_macros.h"

#if TEST_CUDA_COMPILER(NVCC, <, 12, 5)
#  define TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR // VSO#424280
#endif

#if TEST_COMPILER(MSVC)
#  ifndef _MSC_EXTENSIONS
#    define TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK // VSO#119998
#  endif
#endif

#if TEST_COMPILER(GCC, <, 9)
#  define TEST_WORKAROUND_CONSTEXPR_IMPLIES_NOEXCEPT // GCC PR 87603
#endif // TEST_COMPILER(GCC, <, 9)

#endif // SUPPORT_TEST_WORKAROUNDS_H
