/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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

#include "ExceptionDetails.h"
#include "ExceptionOr.h"
#include <JavaScriptCore/ThrowScope.h>

namespace JSC {
class CatchScope;
}

namespace WebCore {

class CachedScript;
class DeferredPromise;
class JSDOMGlobalObject;

void throwAttributeTypeError(JSC::JSGlobalObject&, JSC::ThrowScope&, ASCIILiteral interfaceName, ASCIILiteral attributeName, ASCIILiteral expectedType);

void throwDataCloneError(JSC::JSGlobalObject&, JSC::ThrowScope&);
void throwDOMSyntaxError(JSC::JSGlobalObject&, JSC::ThrowScope&, ASCIILiteral); // Not the same as a JavaScript syntax error.
void throwInvalidStateError(JSC::JSGlobalObject&, JSC::ThrowScope&, ASCIILiteral);
WEBCORE_EXPORT void throwNonFiniteTypeError(JSC::JSGlobalObject&, JSC::ThrowScope&);
void throwNotSupportedError(JSC::JSGlobalObject&, JSC::ThrowScope&, ASCIILiteral);
void throwSecurityError(JSC::JSGlobalObject&, JSC::ThrowScope&, const String& message);
WEBCORE_EXPORT void throwSequenceTypeError(JSC::JSGlobalObject&, JSC::ThrowScope&);

WEBCORE_EXPORT JSC::EncodedJSValue throwArgumentMustBeEnumError(JSC::JSGlobalObject&, JSC::ThrowScope&, unsigned argumentIndex, ASCIILiteral argumentName, ASCIILiteral functionInterfaceName, ASCIILiteral functionName, ASCIILiteral expectedValues);
WEBCORE_EXPORT JSC::EncodedJSValue throwArgumentMustBeFunctionError(JSC::JSGlobalObject&, JSC::ThrowScope&, unsigned argumentIndex, ASCIILiteral argumentName, ASCIILiteral functionInterfaceName, ASCIILiteral functionName);
WEBCORE_EXPORT JSC::EncodedJSValue throwArgumentMustBeObjectError(JSC::JSGlobalObject&, JSC::ThrowScope&, unsigned argumentIndex, ASCIILiteral argumentName, ASCIILiteral functionInterfaceName, ASCIILiteral functionName);
WEBCORE_EXPORT JSC::EncodedJSValue throwArgumentTypeError(JSC::JSGlobalObject&, JSC::ThrowScope&, unsigned argumentIndex, ASCIILiteral argumentName, ASCIILiteral functionInterfaceName, ASCIILiteral functionName, ASCIILiteral expectedType);
WEBCORE_EXPORT JSC::EncodedJSValue throwRequiredMemberTypeError(JSC::JSGlobalObject&, JSC::ThrowScope&, ASCIILiteral memberName, ASCIILiteral dictionaryName, ASCIILiteral expectedType);
JSC::EncodedJSValue throwConstructorScriptExecutionContextUnavailableError(JSC::JSGlobalObject&, JSC::ThrowScope&, ASCIILiteral interfaceName);

String makeThisTypeErrorMessage(const char* interfaceName, const char* attributeName);
String makeUnsupportedIndexedSetterErrorMessage(ASCIILiteral interfaceName);

WEBCORE_EXPORT JSC::EncodedJSValue throwThisTypeError(JSC::JSGlobalObject&, JSC::ThrowScope&, const char* interfaceName, const char* functionName);

WEBCORE_EXPORT JSC::EncodedJSValue rejectPromiseWithGetterTypeError(JSC::JSGlobalObject&, const JSC::ClassInfo*, JSC::PropertyName attributeName);
WEBCORE_EXPORT JSC::EncodedJSValue rejectPromiseWithThisTypeError(DeferredPromise&, const char* interfaceName, const char* operationName);
WEBCORE_EXPORT JSC::EncodedJSValue rejectPromiseWithThisTypeError(JSC::JSGlobalObject&, const char* interfaceName, const char* operationName);

String retrieveErrorMessageWithoutName(JSC::JSGlobalObject&, JSC::VM&, JSC::JSValue exception, JSC::CatchScope&);
String retrieveErrorMessage(JSC::JSGlobalObject&, JSC::VM&, JSC::JSValue exception, JSC::CatchScope&);
WEBCORE_EXPORT void reportException(JSC::JSGlobalObject*, JSC::JSValue exception, CachedScript* = nullptr, bool = false);
WEBCORE_EXPORT void reportException(JSC::JSGlobalObject*, JSC::Exception*, CachedScript* = nullptr, bool = false, ExceptionDetails* = nullptr);
WEBCORE_EXPORT void reportExceptionIfJSDOMWindow(JSC::JSGlobalObject*, JSC::JSValue exception);
void reportCurrentException(JSC::JSGlobalObject*);

JSC::JSValue createDOMException(JSC::JSGlobalObject&, Exception&&);
JSC::JSValue createDOMException(JSC::JSGlobalObject*, ExceptionCode, const String& = emptyString());

// Convert a DOM implementation exception into a JavaScript exception in the execution lexicalGlobalObject.
WEBCORE_EXPORT void propagateExceptionSlowPath(JSC::JSGlobalObject&, JSC::ThrowScope&, Exception&&);

ALWAYS_INLINE void propagateException(JSC::JSGlobalObject& lexicalGlobalObject, JSC::ThrowScope& throwScope, Exception&& exception)
{
    if (throwScope.exception())
        return;
    propagateExceptionSlowPath(lexicalGlobalObject, throwScope, WTFMove(exception));
}

inline void propagateException(JSC::JSGlobalObject& lexicalGlobalObject, JSC::ThrowScope& throwScope, ExceptionOr<void>&& value)
{
    if (UNLIKELY(value.hasException()))
        propagateException(lexicalGlobalObject, throwScope, value.releaseException());
}

template<typename Functor> void invokeFunctorPropagatingExceptionIfNecessary(JSC::JSGlobalObject& lexicalGlobalObject, JSC::ThrowScope& throwScope, NOESCAPE Functor&& functor)
{
    using ReturnType = std::invoke_result_t<Functor>;

    if constexpr (IsExceptionOr<ReturnType>) {
        auto result = functor();
        if (UNLIKELY(result.hasException()))
            propagateException(lexicalGlobalObject, throwScope, result.releaseException());
    } else
        functor();
}

} // namespace WebCore
