/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include <sys/cdefs.h>
#include <sys/types.h>
#include "fenv.h"

#define ROUND_MASK   (FE_TONEAREST | FE_DOWNWARD | FE_UPWARD | FE_TOWARDZERO)

/*
 * The hardware default control word for i387's and later coprocessors is
 * 0x37F, giving:
 *
 *	round to nearest
 *	64-bit precision
 *	all exceptions masked.
 *
 * We modify the affine mode bit and precision bits in this to give:
 *
 *	affine mode for 287's (if they work at all) (1 in bitfield 1<<12)
 *	53-bit precision (2 in bitfield 3<<8)
 *
 * 64-bit precision often gives bad results with high level languages
 * because it makes the results of calculations depend on whether
 * intermediate values are stored in memory or in FPU registers.
 */
#define	__INITIAL_NPXCW__	0x127F
#define	__INITIAL_MXCSR__	0x1F80

/*
 * As compared to the x87 control word, the SSE unit's control word
 * has the rounding control bits offset by 3 and the exception mask
 * bits offset by 7.
 */
#define _SSE_ROUND_SHIFT 3
#define _SSE_EMASK_SHIFT 7

const fenv_t __fe_dfl_env = {
  __INITIAL_NPXCW__, /*__control*/
  0x0000,            /*__mxcsr_hi*/
  0x0000,            /*__status*/
  0x1f80,            /*__mxcsr_lo*/
  0xffffffff,        /*__tag*/
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff } /*__other*/
};

#define __fldcw(__cw)           __asm volatile("fldcw %0" : : "m" (__cw))
#define __fldenv(__env)         __asm volatile("fldenv %0" : : "m" (__env))
#define __fldenvx(__env)        __asm volatile("fldenv %0" : : "m" (__env)  \
                                : "st", "st(1)", "st(2)", "st(3)", "st(4)",   \
                                "st(5)", "st(6)", "st(7)")
#define __fnclex()              __asm volatile("fnclex")
#define __fnstenv(__env)        __asm volatile("fnstenv %0" : "=m" (*(__env)))
#define __fnstcw(__cw)          __asm volatile("fnstcw %0" : "=m" (*(__cw)))
#define __fnstsw(__sw)          __asm volatile("fnstsw %0" : "=am" (*(__sw)))
#define __fwait()               __asm volatile("fwait")
#define __ldmxcsr(__csr)        __asm volatile("ldmxcsr %0" : : "m" (__csr))
#define __stmxcsr(__csr)        __asm volatile("stmxcsr %0" : "=m" (*(__csr)))

/* After testing for SSE support once, we cache the result in __has_sse. */
enum __sse_support { __SSE_YES, __SSE_NO, __SSE_UNK };
#ifdef __SSE__
#define __HAS_SSE()     1
#else
#define __HAS_SSE()     (__has_sse == __SSE_YES ||                      \
                        (__has_sse == __SSE_UNK && __test_sse()))
#endif

enum __sse_support __has_sse =
#ifdef __SSE__
  __SSE_YES;
#else
  __SSE_UNK;
#endif

#ifndef __SSE__
#define getfl(x)    __asm volatile("pushfl\n\tpopl %0" : "=mr" (*(x)))
#define setfl(x)    __asm volatile("pushl %0\n\tpopfl" : : "g" (x))
#define cpuid_dx(x) __asm volatile("pushl %%ebx\n\tmovl $1, %%eax\n\t"  \
                    "cpuid\n\tpopl %%ebx"          \
                    : "=d" (*(x)) : : "eax", "ecx")

/*
 * Test for SSE support on this processor.  We need to do this because
 * we need to use ldmxcsr/stmxcsr to get correct results if any part
 * of the program was compiled to use SSE floating-point, but we can't
 * use SSE on older processors.
 */
int
__test_sse(void)
{
  int flag, nflag;
  int dx_features;

  /* Am I a 486? */
  getfl(&flag);
  nflag = flag ^ 0x200000;
  setfl(nflag);
  getfl(&nflag);
  if (flag != nflag) {
    /* Not a 486, so CPUID should work. */
    cpuid_dx(&dx_features);
    if (dx_features & 0x2000000) {
      __has_sse = __SSE_YES;
      return (1);
    }
  }
  __has_sse = __SSE_NO;
  return (0);
}
#endif /* __SSE__ */

int
fesetexceptflag(const fexcept_t *flagp, int excepts)
{
  fenv_t env;
  __uint32_t mxcsr;

  excepts &= FE_ALL_EXCEPT;
  if (excepts) { /* Do nothing if excepts is 0 */
    __fnstenv(&env);
    env.__status &= ~excepts;
    env.__status |= *flagp & excepts;
    __fnclex();
    __fldenv(env);
    if (__HAS_SSE()) {
      __stmxcsr(&mxcsr);
      mxcsr &= ~excepts;
      mxcsr |= *flagp & excepts;
      __ldmxcsr(mxcsr);
    }
  }

  return (0);
}

int
feraiseexcept(int excepts)
{
  fexcept_t ex = excepts;

  fesetexceptflag(&ex, excepts);
  __fwait();
  return (0);
}

int
fegetenv(fenv_t *envp)
{
  __uint32_t mxcsr;

  __fnstenv(envp);
  /*
   * fnstenv masks all exceptions, so we need to restore
   * the old control word to avoid this side effect.
   */
  __fldcw(envp->__control);
  if (__HAS_SSE()) {
    __stmxcsr(&mxcsr);
    envp->__mxcsr_hi = mxcsr >> 16;
    envp->__mxcsr_lo = mxcsr & 0xffff;
  }
  return (0);
}

int
feholdexcept(fenv_t *envp)
{
  __uint32_t mxcsr;
  fenv_t env;

  __fnstenv(&env);
  *envp = env;
  env.__status &= ~FE_ALL_EXCEPT;
  env.__control |= FE_ALL_EXCEPT;
  __fnclex();
  __fldenv(env);
  if (__HAS_SSE()) {
    __stmxcsr(&mxcsr);
    envp->__mxcsr_hi = mxcsr >> 16;
    envp->__mxcsr_lo = mxcsr & 0xffff;
    mxcsr &= ~FE_ALL_EXCEPT;
    mxcsr |= FE_ALL_EXCEPT << _SSE_EMASK_SHIFT;
    __ldmxcsr(mxcsr);
  }
  return (0);
}

int
feupdateenv(const fenv_t *envp)
{
  __uint32_t mxcsr;
  __uint16_t status;

  __fnstsw(&status);
  if (__HAS_SSE()) {
    __stmxcsr(&mxcsr);
  } else {
    mxcsr = 0;
  }
  fesetenv(envp);
  feraiseexcept((mxcsr | status) & FE_ALL_EXCEPT);
  return (0);
}

int
feenableexcept(int mask)
{
  __uint32_t mxcsr;
  __uint16_t control, omask;

  mask &= FE_ALL_EXCEPT;
  __fnstcw(&control);
  if (__HAS_SSE()) {
    __stmxcsr(&mxcsr);
  } else {
    mxcsr = 0;
  }
  omask = ~(control | mxcsr >> _SSE_EMASK_SHIFT) & FE_ALL_EXCEPT;
  if (mask) {
    control &= ~mask;
    __fldcw(control);
    if (__HAS_SSE()) {
      mxcsr &= ~(mask << _SSE_EMASK_SHIFT);
      __ldmxcsr(mxcsr);
    }
  }
  return (omask);
}

int
fedisableexcept(int mask)
{
  __uint32_t mxcsr;
  __uint16_t control, omask;

  mask &= FE_ALL_EXCEPT;
  __fnstcw(&control);
  if (__HAS_SSE()) {
    __stmxcsr(&mxcsr);
  } else {
    mxcsr = 0;
  }
  omask = ~(control | mxcsr >> _SSE_EMASK_SHIFT) & FE_ALL_EXCEPT;
  if (mask) {
    control |= mask;
    __fldcw(control);
    if (__HAS_SSE()) {
      mxcsr |= mask << _SSE_EMASK_SHIFT;
      __ldmxcsr(mxcsr);
    }
  }
  return (omask);
}

int
feclearexcept(int excepts)
{
  fenv_t env;
  __uint32_t mxcsr;

  excepts &= FE_ALL_EXCEPT;
  if (excepts) { /* Do nothing if excepts is 0 */
    __fnstenv(&env);
    env.__status &= ~excepts;
    __fnclex();
    __fldenv(env);
    if (__HAS_SSE()) {
      __stmxcsr(&mxcsr);
      mxcsr &= ~excepts;
      __ldmxcsr(mxcsr);
    }
  }
  return (0);
}

int
fegetexceptflag(fexcept_t *flagp, int excepts)
{
  __uint32_t mxcsr;
  __uint16_t status;

  excepts &= FE_ALL_EXCEPT;
  __fnstsw(&status);
  if (__HAS_SSE()) {
    __stmxcsr(&mxcsr);
  } else {
    mxcsr = 0;
  }
  *flagp = (status | mxcsr) & excepts;
  return (0);
}

int
fetestexcept(int excepts)
{
  __uint32_t mxcsr;
  __uint16_t status;

  excepts &= FE_ALL_EXCEPT;
  if (excepts) { /* Do nothing if excepts is 0 */
    __fnstsw(&status);
    if (__HAS_SSE()) {
      __stmxcsr(&mxcsr);
    } else {
      mxcsr = 0;
    }
    return ((status | mxcsr) & excepts);
  }
  return (0);
}

int
fegetround(void)
{
  __uint16_t control;

  /*
   * We assume that the x87 and the SSE unit agree on the
   * rounding mode.  Reading the control word on the x87 turns
   * out to be about 5 times faster than reading it on the SSE
   * unit on an Opteron 244.
   */
  __fnstcw(&control);
  return (control & ROUND_MASK);
}

int
fesetround(int round)
{
  __uint32_t mxcsr;
  __uint16_t control;

  if (round & ~ROUND_MASK) {
    return (-1);
  } else {
    __fnstcw(&control);
    control &= ~ROUND_MASK;
    control |= round;
    __fldcw(control);
    if (__HAS_SSE()) {
      __stmxcsr(&mxcsr);
      mxcsr &= ~(ROUND_MASK << _SSE_ROUND_SHIFT);
      mxcsr |= round << _SSE_ROUND_SHIFT;
      __ldmxcsr(mxcsr);
    }
    return (0);
  }
}

int
fesetenv(const fenv_t *envp)
{
  fenv_t env = *envp;
  __uint32_t mxcsr;

  mxcsr = (env.__mxcsr_hi << 16) | (env.__mxcsr_lo);
  env.__mxcsr_hi = 0xffff;
  env.__mxcsr_lo = 0xffff;
  /*
   * XXX Using fldenvx() instead of fldenv() tells the compiler that this
   * instruction clobbers the i387 register stack.  This happens because
   * we restore the tag word from the saved environment.  Normally, this
   * would happen anyway and we wouldn't care, because the ABI allows
   * function calls to clobber the i387 regs.  However, fesetenv() is
   * inlined, so we need to be more careful.
   */
  __fldenvx(env);
  if (__HAS_SSE()) {
    __ldmxcsr(mxcsr);
  }
  return (0);
}

int
fegetexcept(void)
{
  __uint16_t control;

  /*
   * We assume that the masks for the x87 and the SSE unit are
   * the same.
   */
  __fnstcw(&control);
  return (~control & FE_ALL_EXCEPT);
}
