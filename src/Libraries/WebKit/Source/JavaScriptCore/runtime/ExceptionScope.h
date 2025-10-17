/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#pragma once

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include "VM.h"
#include <wtf/StackPointer.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

namespace JSC {
    
class Exception;
    
#if ENABLE(EXCEPTION_SCOPE_VERIFICATION)

#define EXCEPTION_ASSERT(assertion) RELEASE_ASSERT(assertion)
#define EXCEPTION_ASSERT_UNUSED(variable, assertion) RELEASE_ASSERT(assertion)
#define EXCEPTION_ASSERT_WITH_MESSAGE(assertion, message) RELEASE_ASSERT_WITH_MESSAGE(assertion, message)

#if ENABLE(C_LOOP)
#define EXCEPTION_SCOPE_POSITION_FOR_ASAN(vm__) (vm__).currentCLoopStackPointer()
#elif ASAN_ENABLED
#define EXCEPTION_SCOPE_POSITION_FOR_ASAN(vm__) currentStackPointer()
#else
#define EXCEPTION_SCOPE_POSITION_FOR_ASAN(vm__) nullptr
#endif

class ExceptionScope {
public:
    VM& vm() const { return m_vm; }
    unsigned recursionDepth() const { return m_recursionDepth; }
    ALWAYS_INLINE Exception* exception() const { return m_vm.exception(); }

    ALWAYS_INLINE void assertNoException() { RELEASE_ASSERT_WITH_MESSAGE(!exception(), "%s", unexpectedExceptionMessage().data()); }
    ALWAYS_INLINE void releaseAssertNoException() { RELEASE_ASSERT_WITH_MESSAGE(!exception(), "%s", unexpectedExceptionMessage().data()); }
    ALWAYS_INLINE void assertNoExceptionExceptTermination() { RELEASE_ASSERT_WITH_MESSAGE(!exception() || m_vm.hasPendingTerminationException(), "%s", unexpectedExceptionMessage().data()); }
    ALWAYS_INLINE void releaseAssertNoExceptionExceptTermination() { RELEASE_ASSERT_WITH_MESSAGE(!exception() || m_vm.hasPendingTerminationException(), "%s", unexpectedExceptionMessage().data()); }

#if ASAN_ENABLED || ENABLE(C_LOOP)
    const void* stackPosition() const {  return m_location.stackPosition; }
#else
    const void* stackPosition() const {  return this; }
#endif

protected:
    ExceptionScope(VM&, ExceptionEventLocation);
    ExceptionScope(const ExceptionScope&) = delete;
    ExceptionScope(ExceptionScope&&) = default;
    ~ExceptionScope();

    JS_EXPORT_PRIVATE CString unexpectedExceptionMessage();

    VM& m_vm;
    ExceptionScope* m_previousScope;
    ExceptionEventLocation m_location;
    unsigned m_recursionDepth;
};

#else // not ENABLE(EXCEPTION_SCOPE_VERIFICATION)
    
#define EXCEPTION_ASSERT(x) ASSERT(x)
#define EXCEPTION_ASSERT_UNUSED(variable, assertion) ASSERT_UNUSED(variable, assertion)
#define EXCEPTION_ASSERT_WITH_MESSAGE(assertion, message) ASSERT_WITH_MESSAGE(assertion, message)

class ExceptionScope {
public:
    ALWAYS_INLINE VM& vm() const { return m_vm; }
    ALWAYS_INLINE Exception* exception() const { return m_vm.exception(); }

    ALWAYS_INLINE void assertNoException() { ASSERT(!exception()); }
    ALWAYS_INLINE void releaseAssertNoException() { RELEASE_ASSERT(!exception()); }
    ALWAYS_INLINE void assertNoExceptionExceptTermination() { ASSERT(!exception() || m_vm.hasPendingTerminationException()); }
    ALWAYS_INLINE void releaseAssertNoExceptionExceptTermination() { RELEASE_ASSERT(!exception() || m_vm.hasPendingTerminationException()); }

protected:
    ALWAYS_INLINE ExceptionScope(VM& vm)
        : m_vm(vm)
    { }
    ExceptionScope(const ExceptionScope&) = delete;
    ExceptionScope(ExceptionScope&&) = default;

    ALWAYS_INLINE CString unexpectedExceptionMessage() { return { }; }

    VM& m_vm;
};

#endif // ENABLE(EXCEPTION_SCOPE_VERIFICATION)

#define RETURN_IF_EXCEPTION(scope__, value__) do { \
        SUPPRESS_UNCOUNTED_LOCAL JSC::VM& vm = (scope__).vm(); \
        EXCEPTION_ASSERT(!!(scope__).exception() == vm.traps().needHandling(JSC::VMTraps::NeedExceptionHandling)); \
        if (UNLIKELY(vm.traps().maybeNeedHandling())) { \
            if (vm.hasExceptionsAfterHandlingTraps()) \
                return value__; \
        } \
    } while (false)

#define RETURN_IF_EXCEPTION_WITH_TRAPS_DEFERRED(scope__, value__) do { \
        if (UNLIKELY((scope__).exception())) \
            return value__; \
    } while (false)

#define RELEASE_AND_RETURN(scope__, expression__) do { \
        scope__.release(); \
        return expression__; \
    } while (false)

} // namespace JSC
