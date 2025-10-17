/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#include <stdint.h>

const fenv_t __fe_dfl_env = 0;

int fegetenv(fenv_t* envp) {
  __asm__ __volatile__("frcsr %0" : "=r"(*envp));
  return 0;
}

int fesetenv(const fenv_t* envp) {
  fenv_t env;
  fegetenv(&env);
  if (*envp != env) {
    __asm__ __volatile__("fscsr %z0" : : "r"(*envp));
  }
  return 0;
}

int feclearexcept(int excepts) {
  __asm__ __volatile__("csrc fflags, %0" : : "r"(excepts & FE_ALL_EXCEPT));
  return 0;
}

int fegetexceptflag(fexcept_t* flagp, int excepts) {
  *flagp = fetestexcept(excepts & FE_ALL_EXCEPT);
  return 0;
}

int fesetexceptflag(const fexcept_t* flagp, int excepts) {
  feclearexcept((~*flagp) & excepts);
  feraiseexcept(*flagp & excepts);
  return 0;
}

int feraiseexcept(int excepts) {
  __asm__ __volatile__("csrs fflags, %0" : : "r"(excepts));
  return 0;
}

int fetestexcept(int excepts) {
  int flags;
  __asm__ __volatile__("frflags %0" : "=r"(flags));
  return flags & excepts;
}

int fegetround(void) {
  int rm;
  __asm__ __volatile__("frrm %0" : "=r"(rm));
  return rm;
}

int fesetround(int round) {
  if (round < FE_TONEAREST || round > FE_UPWARD) return -1;
  __asm__ __volatile__("fsrm %z0" : : "r"(round));
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
