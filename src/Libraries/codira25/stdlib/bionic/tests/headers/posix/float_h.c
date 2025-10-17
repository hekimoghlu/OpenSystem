/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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

#include <float.h>

#include "header_checks.h"

static void float_h() {
  int flt_rounds = FLT_ROUNDS;

  MACRO(FLT_EVAL_METHOD);

  MACRO(FLT_RADIX);
  MACRO(FLT_MANT_DIG);
  MACRO(DBL_MANT_DIG);
  MACRO(LDBL_MANT_DIG);
  MACRO(DECIMAL_DIG);
  MACRO(FLT_DIG);
  MACRO(DBL_DIG);
  MACRO(LDBL_DIG);
  MACRO(FLT_MIN_EXP);
  MACRO(DBL_MIN_EXP);
  MACRO(LDBL_MIN_EXP);
  MACRO(FLT_MIN_10_EXP);
  MACRO(DBL_MIN_10_EXP);
  MACRO(LDBL_MIN_10_EXP);
  MACRO(FLT_MAX_EXP);
  MACRO(DBL_MAX_EXP);
  MACRO(LDBL_MAX_EXP);
  MACRO(FLT_MAX_10_EXP);
  MACRO(DBL_MAX_10_EXP);
  MACRO(LDBL_MAX_10_EXP);
  MACRO(FLT_MAX);
  MACRO(DBL_MAX);
  MACRO(LDBL_MAX);
  MACRO(FLT_EPSILON);
  MACRO(DBL_EPSILON);
  MACRO(LDBL_EPSILON);
  MACRO(FLT_MIN);
  MACRO(DBL_MIN);
  MACRO(LDBL_MIN);
}
