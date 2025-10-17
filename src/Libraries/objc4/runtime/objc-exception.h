/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#ifndef __OBJC_EXCEPTION_H_
#define __OBJC_EXCEPTION_H_

#include <objc/objc.h>
#include <stdint.h>

typedef id _Nonnull (*objc_exception_preprocessor)(id _Nonnull exception);
typedef int (*objc_exception_matcher)(Class _Nonnull catch_type,
                                      id _Nonnull exception);
typedef void (*objc_uncaught_exception_handler)(id _Null_unspecified /* _Nonnull */ exception);
typedef void (*objc_exception_handler)(id _Nullable unused,
                                       void * _Nullable context);

/** 
 * Throw a runtime exception. This function is inserted by the compiler
 * where \c @throw would otherwise be.
 * 
 * @param exception The exception to be thrown.
 */
OBJC_COLD OBJC_EXPORT OBJC_NORETURN void
objc_exception_throw(id _Nonnull exception)
    OBJC_AVAILABLE(10.5, 2.0, 9.0, 1.0, 2.0);

OBJC_COLD OBJC_EXPORT OBJC_NORETURN void
objc_exception_rethrow(void)
    OBJC_AVAILABLE(10.5, 2.0, 9.0, 1.0, 2.0);

OBJC_EXPORT id _Nonnull
objc_begin_catch(void * _Nonnull exc_buf)
    OBJC_AVAILABLE(10.5, 2.0, 9.0, 1.0, 2.0);

OBJC_EXPORT void
objc_end_catch(void)
    OBJC_AVAILABLE(10.5, 2.0, 9.0, 1.0, 2.0);

OBJC_COLD OBJC_EXPORT OBJC_NORETURN void
objc_terminate(void)
    OBJC_AVAILABLE(10.8, 6.0, 9.0, 1.0, 2.0);

OBJC_EXPORT objc_exception_preprocessor _Nonnull
objc_setExceptionPreprocessor(objc_exception_preprocessor _Nonnull fn)
    OBJC_AVAILABLE(10.5, 2.0, 9.0, 1.0, 2.0);

OBJC_EXPORT objc_exception_matcher _Nonnull
objc_setExceptionMatcher(objc_exception_matcher _Nonnull fn)
    OBJC_AVAILABLE(10.5, 2.0, 9.0, 1.0, 2.0);

OBJC_EXPORT objc_uncaught_exception_handler _Nonnull
objc_setUncaughtExceptionHandler(objc_uncaught_exception_handler _Nonnull fn)
    OBJC_AVAILABLE(10.5, 2.0, 9.0, 1.0, 2.0);

#if !TARGET_OS_EXCLAVEKIT
// Not for iOS.
OBJC_EXPORT uintptr_t
objc_addExceptionHandler(objc_exception_handler _Nonnull fn,
                         void * _Nullable context)
    OBJC_OSX_AVAILABLE_OTHERS_UNAVAILABLE(10.5);

OBJC_EXPORT void
objc_removeExceptionHandler(uintptr_t token)
    OBJC_OSX_AVAILABLE_OTHERS_UNAVAILABLE(10.5);
#endif // !TARGET_OS_EXCLAVEKIT

#endif  // __OBJC_EXCEPTION_H_

