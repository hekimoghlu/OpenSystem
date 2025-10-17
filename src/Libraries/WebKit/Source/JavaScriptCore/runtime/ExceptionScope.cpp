/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#include "ExceptionScope.h"

#include "ErrorInstance.h"
#include "Exception.h"
#include <wtf/StackTrace.h>
#include <wtf/StringPrintStream.h>
#include <wtf/Threading.h>

namespace JSC {
    
#if ENABLE(EXCEPTION_SCOPE_VERIFICATION)
    
ExceptionScope::ExceptionScope(VM& vm, ExceptionEventLocation location)
    : m_vm(vm)
    , m_previousScope(vm.m_topExceptionScope)
    , m_location(location)
    , m_recursionDepth(m_previousScope ? m_previousScope->m_recursionDepth + 1 : 0)
{
    m_vm.m_topExceptionScope = this;
}

ExceptionScope::~ExceptionScope()
{
    RELEASE_ASSERT(m_vm.m_topExceptionScope);
    m_vm.m_topExceptionScope = m_previousScope;
}

CString ExceptionScope::unexpectedExceptionMessage()
{
    StringPrintStream out;

    out.println("Unexpected exception observed on thread ", Thread::current(), " at:");
    auto currentStack = StackTrace::captureStackTrace(Options::unexpectedExceptionStackTraceLimit(), 1);
    out.print(StackTracePrinter { *currentStack, "    " });

    if (!m_vm.nativeStackTraceOfLastThrow())
        return CString();
    
    out.println("The exception was thrown from thread ", *m_vm.throwingThread(), " at:");
    out.print(StackTracePrinter { *m_vm.nativeStackTraceOfLastThrow(), "    " });

    if (auto* error = jsDynamicCast<ErrorInstance*>(exception()->value()))
        out.println("Error Exception: ", error->tryGetMessageForDebugging());
    else
        out.println("non-Error Exception: ", exception()->value());

    return out.toCString();
}

#endif // ENABLE(EXCEPTION_SCOPE_VERIFICATION)
    
} // namespace JSC
