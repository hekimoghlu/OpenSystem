/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
/* Add signal handling macros
   Make the global variable and signal handler part of the C-API
*/

#ifndef NPY_INTERRUPT_H
#define NPY_INTERRUPT_H

#ifndef NPY_NO_SIGNAL

#include <setjmp.h>
#include <signal.h>

#ifndef sigsetjmp

#define NPY_SIGSETJMP(arg1, arg2) setjmp(arg1)
#define NPY_SIGLONGJMP(arg1, arg2) longjmp(arg1, arg2)
#define NPY_SIGJMP_BUF jmp_buf

#else

#define NPY_SIGSETJMP(arg1, arg2) sigsetjmp(arg1, arg2)
#define NPY_SIGLONGJMP(arg1, arg2) siglongjmp(arg1, arg2)
#define NPY_SIGJMP_BUF sigjmp_buf

#endif

#    define NPY_SIGINT_ON {                                             \
                   PyOS_sighandler_t _npy_sig_save;                     \
                   _npy_sig_save = PyOS_setsig(SIGINT, _PyArray_SigintHandler); \
                   if (NPY_SIGSETJMP(*((NPY_SIGJMP_BUF *)_PyArray_GetSigintBuf()), \
                                 1) == 0) {                             \

#    define NPY_SIGINT_OFF }                                      \
        PyOS_setsig(SIGINT, _npy_sig_save);                       \
        }

#else /* NPY_NO_SIGNAL  */

#define NPY_SIGINT_ON
#define NPY_SIGINT_OFF

#endif /* HAVE_SIGSETJMP */

#endif /* NPY_INTERRUPT_H */
