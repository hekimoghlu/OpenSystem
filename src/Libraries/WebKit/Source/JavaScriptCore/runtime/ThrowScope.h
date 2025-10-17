/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

class CallFrame;
class JSObject;

#if ENABLE(EXCEPTION_SCOPE_VERIFICATION)

// If a function can throw a JS exception, it should declare a ThrowScope at the
// top of the function (as early as possible) using the DECLARE_THROW_SCOPE macro.
// Declaring a ThrowScope in a function means that the function may throw an
// exception that its caller will have to handle.

class ThrowScope : public ExceptionScope {
public:
    JS_EXPORT_PRIVATE ThrowScope(VM&, ExceptionEventLocation);
    JS_EXPORT_PRIVATE ~ThrowScope();

    ThrowScope(const ThrowScope&) = delete;
    ThrowScope(ThrowScope&&) = default;

    JS_EXPORT_PRIVATE Exception* throwException(JSGlobalObject*, Exception*);
    JS_EXPORT_PRIVATE Exception* throwException(JSGlobalObject*, JSValue);

    void release() { m_isReleased = true; }

    void clearException() { m_vm.clearException(); }

private:
    void simulateThrow();

    bool m_isReleased { false };
};

#define DECLARE_THROW_SCOPE(vm__) \
    JSC::ThrowScope((vm__), JSC::ExceptionEventLocation(EXCEPTION_SCOPE_POSITION_FOR_ASAN(vm__), __FUNCTION__, __FILE__, __LINE__))

#else // not ENABLE(EXCEPTION_SCOPE_VERIFICATION)

class ThrowScope : public ExceptionScope {
public:
    ALWAYS_INLINE ThrowScope(VM& vm)
        : ExceptionScope(vm)
    { }
    ThrowScope(const ThrowScope&) = delete;
    ThrowScope(ThrowScope&&) = default;

    ALWAYS_INLINE Exception* throwException(JSGlobalObject* globalObject, Exception* exception) { return m_vm.throwException(globalObject, exception); }
    ALWAYS_INLINE Exception* throwException(JSGlobalObject* globalObject, JSValue value) { return m_vm.throwException(globalObject, value); }

    ALWAYS_INLINE void release() { }

    ALWAYS_INLINE void clearException() { m_vm.clearException(); }
};

#define DECLARE_THROW_SCOPE(vm__) \
    JSC::ThrowScope((vm__))

#endif // ENABLE(EXCEPTION_SCOPE_VERIFICATION)

ALWAYS_INLINE Exception* throwException(JSGlobalObject* globalObject, ThrowScope& scope, Exception* exception)
{
    return scope.throwException(globalObject, exception);
}

ALWAYS_INLINE Exception* throwException(JSGlobalObject* globalObject, ThrowScope& scope, JSValue value)
{
    return scope.throwException(globalObject, value);
}

ALWAYS_INLINE EncodedJSValue throwVMException(JSGlobalObject* globalObject, ThrowScope& scope, Exception* exception)
{
    throwException(globalObject, scope, exception);
    return encodedJSValue();
}

ALWAYS_INLINE EncodedJSValue throwVMException(JSGlobalObject* globalObject, ThrowScope& scope, JSValue value)
{
    throwException(globalObject, scope, value);
    return encodedJSValue();
}

} // namespace JSC
