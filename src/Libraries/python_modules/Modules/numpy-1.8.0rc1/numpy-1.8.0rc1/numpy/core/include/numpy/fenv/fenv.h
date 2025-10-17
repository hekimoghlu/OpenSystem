/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#ifndef _FENV_H_
#define _FENV_H_

#include <sys/cdefs.h>
#include <sys/types.h>

typedef struct {
        __uint32_t      __control;
        __uint32_t      __status;
        __uint32_t      __tag;
        char            __other[16];
} fenv_t;

typedef __uint16_t      fexcept_t;

/* Exception flags */
#define FE_INVALID      0x01
#define FE_DENORMAL     0x02
#define FE_DIVBYZERO    0x04
#define FE_OVERFLOW     0x08
#define FE_UNDERFLOW    0x10
#define FE_INEXACT      0x20
#define FE_ALL_EXCEPT   (FE_DIVBYZERO | FE_DENORMAL | FE_INEXACT | \
                         FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW)

/* Rounding modes */
#define FE_TONEAREST    0x0000
#define FE_DOWNWARD     0x0400
#define FE_UPWARD       0x0800
#define FE_TOWARDZERO   0x0c00
#define _ROUND_MASK     (FE_TONEAREST | FE_DOWNWARD | \
                         FE_UPWARD | FE_TOWARDZERO)

__BEGIN_DECLS

/* Default floating-point environment */
extern const fenv_t     npy__fe_dfl_env;
#define FE_DFL_ENV      (&npy__fe_dfl_env)

#define __fldcw(__cw)           __asm __volatile("fldcw %0" : : "m" (__cw))
#define __fldenv(__env)         __asm __volatile("fldenv %0" : : "m" (__env))
#define __fnclex()              __asm __volatile("fnclex")
#define __fnstenv(__env)        __asm __volatile("fnstenv %0" : "=m" (*(__env)))
#define __fnstcw(__cw)          __asm __volatile("fnstcw %0" : "=m" (*(__cw)))
#define __fnstsw(__sw)          __asm __volatile("fnstsw %0" : "=am" (*(__sw)))
#define __fwait()               __asm __volatile("fwait")

static __inline int
feclearexcept(int __excepts)
{
        fenv_t __env;

        if (__excepts == FE_ALL_EXCEPT) {
                __fnclex();
        } else {
                __fnstenv(&__env);
                __env.__status &= ~__excepts;
                __fldenv(__env);
        }
        return (0);
}

static __inline int
fegetexceptflag(fexcept_t *__flagp, int __excepts)
{
        __uint16_t __status;

        __fnstsw(&__status);
        *__flagp = __status & __excepts;
        return (0);
}

static __inline int
fesetexceptflag(const fexcept_t *__flagp, int __excepts)
{
        fenv_t __env;

        __fnstenv(&__env);
        __env.__status &= ~__excepts;
        __env.__status |= *__flagp & __excepts;
        __fldenv(__env);
        return (0);
}

static __inline int
feraiseexcept(int __excepts)
{
        fexcept_t __ex = __excepts;

        fesetexceptflag(&__ex, __excepts);
        __fwait();
        return (0);
}

static __inline int
fetestexcept(int __excepts)
{
        __uint16_t __status;

        __fnstsw(&__status);
        return (__status & __excepts);
}

static __inline int
fegetround(void)
{
        int __control;

        __fnstcw(&__control);
        return (__control & _ROUND_MASK);
}

static __inline int
fesetround(int __round)
{
        int __control;

        if (__round & ~_ROUND_MASK)
                return (-1);
        __fnstcw(&__control);
        __control &= ~_ROUND_MASK;
        __control |= __round;
        __fldcw(__control);
        return (0);
}

static __inline int
fegetenv(fenv_t *__envp)
{
        int __control;

        /*
         * fnstenv masks all exceptions, so we need to save and
         * restore the control word to avoid this side effect.
         */
        __fnstcw(&__control);
        __fnstenv(__envp);
        __fldcw(__control);
        return (0);
}

static __inline int
feholdexcept(fenv_t *__envp)
{

        __fnstenv(__envp);
        __fnclex();
        return (0);
}

static __inline int
fesetenv(const fenv_t *__envp)
{

        __fldenv(*__envp);
        return (0);
}

static __inline int
feupdateenv(const fenv_t *__envp)
{
        __uint16_t __status;

        __fnstsw(&__status);
        __fldenv(*__envp);
        feraiseexcept(__status & FE_ALL_EXCEPT);
        return (0);
}

#if __BSD_VISIBLE

static __inline int
fesetmask(int __mask)
{
        int __control;

        __fnstcw(&__control);
        __mask = (__control | FE_ALL_EXCEPT) & ~__mask;
        __fldcw(__mask);
        return (~__control & FE_ALL_EXCEPT);
}

static __inline int
fegetmask(void)
{
        int __control;

        __fnstcw(&__control);
        return (~__control & FE_ALL_EXCEPT);
}

#endif /* __BSD_VISIBLE */

__END_DECLS

#endif  /* !_FENV_H_ */
