/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
#include "JSTestInheritedDictionary2.h"

#include "JSDOMConvertBoolean.h"
#include "JSDOMConvertCallbacks.h"
#include "JSDOMConvertStrings.h"
#include "JSDOMGlobalObject.h"
#include "JSVoidCallback.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/ObjectConstructor.h>



namespace WebCore {
using namespace JSC;

template<> ConversionResult<IDLDictionary<TestInheritedDictionary2>> convertDictionary<TestInheritedDictionary2>(JSGlobalObject& lexicalGlobalObject, JSValue value)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    bool isNullOrUndefined = value.isUndefinedOrNull();
    auto* object = isNullOrUndefined ? nullptr : value.getObject();
    if (UNLIKELY(!isNullOrUndefined && !object)) {
        throwTypeError(&lexicalGlobalObject, throwScope);
        return ConversionResultException { };
    }
    TestInheritedDictionary2 result;
    JSValue boolMemberValue;
    if (isNullOrUndefined)
        boolMemberValue = jsUndefined();
    else {
        boolMemberValue = object->get(&lexicalGlobalObject, Identifier::fromString(vm, "boolMember"_s));
        RETURN_IF_EXCEPTION(throwScope, ConversionResultException { });
    }
    if (!boolMemberValue.isUndefined()) {
        auto boolMemberConversionResult = convert<IDLBoolean>(lexicalGlobalObject, boolMemberValue);
        if (UNLIKELY(boolMemberConversionResult.hasException(throwScope)))
            return ConversionResultException { };
        result.boolMember = boolMemberConversionResult.releaseReturnValue();
    }
    JSValue callbackMemberValue;
    if (isNullOrUndefined)
        callbackMemberValue = jsUndefined();
    else {
        callbackMemberValue = object->get(&lexicalGlobalObject, Identifier::fromString(vm, "callbackMember"_s));
        RETURN_IF_EXCEPTION(throwScope, ConversionResultException { });
    }
    if (!callbackMemberValue.isUndefined()) {
        auto callbackMemberConversionResult = convert<IDLCallbackFunction<JSVoidCallback>>(lexicalGlobalObject, callbackMemberValue, *jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject));
        if (UNLIKELY(callbackMemberConversionResult.hasException(throwScope)))
            return ConversionResultException { };
        result.callbackMember = callbackMemberConversionResult.releaseReturnValue();
    }
    JSValue stringMemberValue;
    if (isNullOrUndefined)
        stringMemberValue = jsUndefined();
    else {
        stringMemberValue = object->get(&lexicalGlobalObject, Identifier::fromString(vm, "stringMember"_s));
        RETURN_IF_EXCEPTION(throwScope, ConversionResultException { });
    }
    if (!stringMemberValue.isUndefined()) {
        auto stringMemberConversionResult = convert<IDLDOMString>(lexicalGlobalObject, stringMemberValue);
        if (UNLIKELY(stringMemberConversionResult.hasException(throwScope)))
            return ConversionResultException { };
        result.stringMember = stringMemberConversionResult.releaseReturnValue();
    }
    return result;
}

JSC::JSObject* convertDictionaryToJS(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const TestInheritedDictionary2& dictionary)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    auto result = constructEmptyObject(&lexicalGlobalObject, globalObject.objectPrototype());

    if (!IDLBoolean::isNullValue(dictionary.boolMember)) {
        auto boolMemberValue = toJS<IDLBoolean>(lexicalGlobalObject, throwScope, IDLBoolean::extractValueFromNullable(dictionary.boolMember));
        RETURN_IF_EXCEPTION(throwScope, { });
        result->putDirect(vm, JSC::Identifier::fromString(vm, "boolMember"_s), boolMemberValue);
    }
    if (!IDLCallbackFunction<JSVoidCallback>::isNullValue(dictionary.callbackMember)) {
        auto callbackMemberValue = toJS<IDLCallbackFunction<JSVoidCallback>>(lexicalGlobalObject, globalObject, throwScope, IDLCallbackFunction<JSVoidCallback>::extractValueFromNullable(dictionary.callbackMember));
        RETURN_IF_EXCEPTION(throwScope, { });
        result->putDirect(vm, JSC::Identifier::fromString(vm, "callbackMember"_s), callbackMemberValue);
    }
    if (!IDLDOMString::isNullValue(dictionary.stringMember)) {
        auto stringMemberValue = toJS<IDLDOMString>(lexicalGlobalObject, throwScope, IDLDOMString::extractValueFromNullable(dictionary.stringMember));
        RETURN_IF_EXCEPTION(throwScope, { });
        result->putDirect(vm, JSC::Identifier::fromString(vm, "stringMember"_s), stringMemberValue);
    }
    return result;
}

} // namespace WebCore

