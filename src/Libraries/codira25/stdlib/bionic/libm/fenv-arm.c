/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#include <fenv.h>

#define FPSCR_RMODE_SHIFT 22

const fenv_t __fe_dfl_env = 0;

int fegetenv(fenv_t* __envp) {
  fenv_t _fpscr;
  __asm__ __volatile__("vmrs %0,fpscr" : "=r"(_fpscr));
  *__envp = _fpscr;
  return 0;
}

int fesetenv(const fenv_t* __envp) {
  fenv_t _fpscr = *__envp;
  __asm__ __volatile__("vmsr fpscr,%0" : : "ri"(_fpscr));
  return 0;
}

int feclearexcept(int __excepts) {
  fexcept_t __fpscr;
  fegetenv(&__fpscr);
  __fpscr &= ~__excepts;
  fesetenv(&__fpscr);
  return 0;
}

int fegetexceptflag(fexcept_t* __flagp, int __excepts) {
  fexcept_t __fpscr;
  fegetenv(&__fpscr);
  *__flagp = __fpscr & __excepts;
  return 0;
}

int fesetexceptflag(const fexcept_t* __flagp, int __excepts) {
  fexcept_t __fpscr;
  fegetenv(&__fpscr);
  __fpscr &= ~__excepts;
  __fpscr |= *__flagp & __excepts;
  fesetenv(&__fpscr);
  return 0;
}

int feraiseexcept(int __excepts) {
  fexcept_t __ex = __excepts;
  fesetexceptflag(&__ex, __excepts);
  return 0;
}

int fetestexcept(int __excepts) {
  fexcept_t __fpscr;
  fegetenv(&__fpscr);
  return (__fpscr & __excepts);
}

int fegetround(void) {
  fenv_t _fpscr;
  fegetenv(&_fpscr);
  return ((_fpscr >> FPSCR_RMODE_SHIFT) & 0x3);
}

int fesetround(int __round) {
  fenv_t _fpscr;
  fegetenv(&_fpscr);
  _fpscr &= ~(0x3 << FPSCR_RMODE_SHIFT);
  _fpscr |= (__round << FPSCR_RMODE_SHIFT);
  fesetenv(&_fpscr);
  return 0;
}

int feholdexcept(fenv_t* __envp) {
  fenv_t __env;
  fegetenv(&__env);
  *__envp = __env;
  __env &= ~FE_ALL_EXCEPT;
  fesetenv(&__env);
  return 0;
}

int feupdateenv(const fenv_t* __envp) {
  fexcept_t __fpscr;
  fegetenv(&__fpscr);
  fesetenv(__envp);
  feraiseexcept(__fpscr & FE_ALL_EXCEPT);
  return 0;
}

int feenableexcept(int __mask __unused) {
  return -1;
}

int fedisableexcept(int __mask __unused) {
  return 0;
}

int fegetexcept(void) {
  return 0;
}
