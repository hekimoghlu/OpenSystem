/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
#include <stdint.h>
#include <fenv.h>

#define FPCR_RMODE_SHIFT 22

const fenv_t __fe_dfl_env = { 0 /* control */, 0 /* status */};

typedef __uint32_t fpu_control_t;   // FPCR, Floating-point Control Register.
typedef __uint32_t fpu_status_t;    // FPSR, Floating-point Status Register.

#define __get(REGISTER, __value) { \
  uint64_t __value64; \
  __asm__ __volatile__("mrs %0," REGISTER : "=r" (__value64)); \
  __value = (__uint32_t) __value64; \
}
#define __get_fpcr(__fpcr) __get("fpcr", __fpcr)
#define __get_fpsr(__fpsr) __get("fpsr", __fpsr)

#define __set(REGISTER, __value) { \
  uint64_t __value64 = __value; \
  __asm__ __volatile__("msr " REGISTER ",%0" : : "ri" (__value64)); \
}
#define __set_fpcr(__fpcr) __set("fpcr", __fpcr)
#define __set_fpsr(__fpsr) __set("fpsr", __fpsr)

int fegetenv(fenv_t* envp) {
  __get_fpcr(envp->__control);
  __get_fpsr(envp->__status);
  return 0;
}

int fesetenv(const fenv_t* envp) {
  fpu_control_t fpcr;
  __get_fpcr(fpcr);
  if (envp->__control != fpcr) {
    __set_fpcr(envp->__control);
  }
  __set_fpsr(envp->__status);
  return 0;
}

int feclearexcept(int excepts) {
  fpu_status_t fpsr;
  __get_fpsr(fpsr);
  fpsr &= ~(excepts & FE_ALL_EXCEPT);
  __set_fpsr(fpsr);
  return 0;
}

int fegetexceptflag(fexcept_t* flagp, int excepts) {
  fpu_status_t fpsr;
  __get_fpsr(fpsr);
  *flagp = fpsr & (excepts & FE_ALL_EXCEPT);
  return 0;
}

int fesetexceptflag(const fexcept_t* flagp, int excepts) {
  excepts &= FE_ALL_EXCEPT;
  fpu_status_t fpsr;
  __get_fpsr(fpsr);
  fpsr &= ~excepts;
  fpsr |= *flagp & excepts;
  __set_fpsr(fpsr);
  return 0;
}

int feraiseexcept(int excepts) {
  fexcept_t ex = excepts;
  fesetexceptflag(&ex, excepts);
  return 0;
}

int fetestexcept(int excepts) {
  fpu_status_t fpsr;
  __get_fpsr(fpsr);
  return (fpsr & (excepts & FE_ALL_EXCEPT));
}

int fegetround(void) {
  fpu_control_t fpcr;
  __get_fpcr(fpcr);
  return ((fpcr >> FPCR_RMODE_SHIFT) & FE_TOWARDZERO);
}

int fesetround(int round) {
  if (round < FE_TONEAREST || round > FE_TOWARDZERO) return -1;
  fpu_control_t fpcr;
  __get_fpcr(fpcr);
  fpu_control_t new_fpcr = fpcr & ~(FE_TOWARDZERO << FPCR_RMODE_SHIFT);
  new_fpcr |= (round << FPCR_RMODE_SHIFT);
  if (new_fpcr != fpcr) {
    __set_fpcr(new_fpcr);
  }
  return 0;
}

int feholdexcept(fenv_t* envp) {
  fegetenv(envp);
  feclearexcept(FE_ALL_EXCEPT);
  return 0;
}

int feupdateenv(const fenv_t* envp) {
  int excepts = fetestexcept(FE_ALL_EXCEPT);
  fesetenv(envp);
  feraiseexcept(excepts);
  return 0;
}

int feenableexcept(int mask __unused) {
  return -1;
}

int fedisableexcept(int mask __unused) {
  return 0;
}

int fegetexcept(void) {
  return 0;
}
