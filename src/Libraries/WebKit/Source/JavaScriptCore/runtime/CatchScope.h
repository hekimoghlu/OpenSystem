/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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

#include "ExceptionScope.h"

namespace JSC {
    
#if ENABLE(EXCEPTION_SCOPE_VERIFICATION)
    
// If a function can clear JS exceptions, it should declare a CatchScope at the
// top of the function (as early as possible) using the DECLARE_CATCH_SCOPE macro.
// Declaring a CatchScope in a function means that the function intends to clear
// pending exceptions before returning to its caller.

class CatchScope : public ExceptionScope {
public:
    JS_EXPORT_PRIVATE CatchScope(VM&, ExceptionEventLocation);
    CatchScope(const CatchScope&) = delete;
    CatchScope(CatchScope&&) = default;

    JS_EXPORT_PRIVATE ~CatchScope();

    void clearException();
    bool clearExceptionExceptTermination();
};

#define DECLARE_CATCH_SCOPE(vm__) \
    JSC::CatchScope((vm__), JSC::ExceptionEventLocation(EXCEPTION_SCOPE_POSITION_FOR_ASAN(vm__), __FUNCTION__, __FILE__, __LINE__))

#else // not ENABLE(EXCEPTION_SCOPE_VERIFICATION)

class CatchScope : public ExceptionScope {
public:
    ALWAYS_INLINE CatchScope(VM& vm)
        : ExceptionScope(vm)
    { }
    CatchScope(const CatchScope&) = delete;
    CatchScope(CatchScope&&) = default;

    ALWAYS_INLINE void clearException();
    ALWAYS_INLINE bool clearExceptionExceptTermination();
};

#define DECLARE_CATCH_SCOPE(vm__) \
    JSC::CatchScope((vm__))

#endif // ENABLE(EXCEPTION_SCOPE_VERIFICATION)

ALWAYS_INLINE void CatchScope::clearException()
{
    m_vm.clearException();
}

ALWAYS_INLINE bool CatchScope::clearExceptionExceptTermination()
{
    if (UNLIKELY(m_vm.hasPendingTerminationException())) {
#if ENABLE(EXCEPTION_SCOPE_VERIFICATION)
        m_vm.exception();
#endif
        return false;
    }
    m_vm.clearException();
    return true;
}

#define CLEAR_AND_RETURN_IF_EXCEPTION(scope__, value__) do { \
        if (UNLIKELY((scope__).exception())) { \
            (scope__).clearException(); \
            return value__; \
        } \
    } while (false)

} // namespace JSC
