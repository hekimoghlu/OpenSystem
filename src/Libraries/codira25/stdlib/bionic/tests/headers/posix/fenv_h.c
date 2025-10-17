/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#include <fenv.h>

#include "header_checks.h"

static void fenv_h() {
  TYPE(fenv_t);
  TYPE(fexcept_t);

  MACRO(FE_DIVBYZERO);
  MACRO(FE_INEXACT);
  MACRO(FE_INVALID);
  MACRO(FE_OVERFLOW);
  MACRO(FE_UNDERFLOW);

  MACRO(FE_ALL_EXCEPT);

  MACRO(FE_DOWNWARD);
  MACRO(FE_TONEAREST);
  MACRO(FE_TOWARDZERO);
  MACRO(FE_UPWARD);

  const fenv_t* fe_dfl_env = FE_DFL_ENV;

  FUNCTION(feclearexcept, int (*f)(int));
  FUNCTION(fegetenv, int (*f)(fenv_t*));
  FUNCTION(fegetexceptflag, int (*f)(fexcept_t*, int));
  FUNCTION(fegetround, int (*f)(void));
  FUNCTION(feholdexcept, int (*f)(fenv_t*));
  FUNCTION(feraiseexcept, int (*f)(int));
  FUNCTION(fesetenv, int (*f)(const fenv_t*));
  FUNCTION(fesetexceptflag, int (*f)(const fexcept_t*, int));
  FUNCTION(fesetround, int (*f)(int));
  FUNCTION(fetestexcept, int (*f)(int));
  FUNCTION(feupdateenv, int (*f)(const fenv_t*));
}
