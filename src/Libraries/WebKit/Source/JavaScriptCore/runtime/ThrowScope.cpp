/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include "config.h"
#include "ThrowScope.h"

#include "Exception.h"
#include "JSCJSValueInlines.h"
#include "VM.h"
#include <wtf/StackTrace.h>

namespace JSC {
    
#if ENABLE(EXCEPTION_SCOPE_VERIFICATION)

ThrowScope::ThrowScope(VM& vm, ExceptionEventLocation location)
    : ExceptionScope(vm, location)
{
    m_vm.verifyExceptionCheckNeedIsSatisfied(m_recursionDepth, m_location);
}

ThrowScope::~ThrowScope()
{
    RELEASE_ASSERT(m_vm.m_topExceptionScope);

    if (!m_isReleased)
        m_vm.verifyExceptionCheckNeedIsSatisfied(m_recursionDepth, m_location);
    else {
        // If we released the scope, that means we're letting our callers do the
        // exception check. However, because our caller may be a LLInt or JIT
        // function (which always checks for exceptions but won't clear the
        // m_needExceptionCheck bit), we should clear m_needExceptionCheck here
        // and let code below decide if we need to simulate a re-throw.
        m_vm.m_needExceptionCheck = false;
    }

    bool willBeHandleByLLIntOrJIT = false;
    const void* previousScopeStackPosition = m_previousScope ? m_previousScope->stackPosition() : nullptr;
    void* topEntryFrame = m_vm.topEntryFrame;

    // If the topEntryFrame was pushed on the stack after the previousScope was instantiated,
    // then this throwScope will be returning to LLINT or JIT code that always do an exception
    // check. In that case, skip the simulated throw because the LLInt and JIT will be
    // checking for the exception their own way instead of calling ThrowScope::exception().
    if (topEntryFrame && previousScopeStackPosition > topEntryFrame)
        willBeHandleByLLIntOrJIT = true;

    if (!willBeHandleByLLIntOrJIT)
        simulateThrow();
}

Exception* ThrowScope::throwException(JSGlobalObject* globalObject, Exception* exception)
{
    if (m_vm.exception() && m_vm.exception() != exception)
        m_vm.verifyExceptionCheckNeedIsSatisfied(m_recursionDepth, m_location);
    
    return m_vm.throwException(globalObject, exception);
}

Exception* ThrowScope::throwException(JSGlobalObject* globalObject, JSValue error)
{
    if (!error.isCell() || error.isObject() || !jsDynamicCast<Exception*>(error.asCell()))
        m_vm.verifyExceptionCheckNeedIsSatisfied(m_recursionDepth, m_location);
    
    return m_vm.throwException(globalObject, error);
}

void ThrowScope::simulateThrow()
{
    RELEASE_ASSERT(m_vm.m_topExceptionScope);
    m_vm.m_simulatedThrowPointLocation = m_location;
    m_vm.m_simulatedThrowPointRecursionDepth = m_recursionDepth;
    m_vm.m_needExceptionCheck = true;
    if (UNLIKELY(Options::dumpSimulatedThrows()))
        m_vm.m_nativeStackTraceOfLastSimulatedThrow = StackTrace::captureStackTrace(Options::unexpectedExceptionStackTraceLimit());
}

#endif // ENABLE(EXCEPTION_SCOPE_VERIFICATION)
    
} // namespace JSC
