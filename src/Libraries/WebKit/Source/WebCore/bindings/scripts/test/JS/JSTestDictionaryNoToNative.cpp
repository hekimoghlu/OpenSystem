/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 13, 2023.
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
#include "JSTestDictionaryNoToNative.h"

#include "JSDOMConvertNumbers.h"
#include "JSDOMGlobalObject.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/ObjectConstructor.h>



namespace WebCore {
using namespace JSC;

JSC::JSObject* convertDictionaryToJS(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const TestDictionaryNoToNative& dictionary)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    auto result = constructEmptyObject(&lexicalGlobalObject, globalObject.objectPrototype());

    if (!IDLDouble::isNullValue(dictionary.member)) {
        auto memberValue = toJS<IDLDouble>(lexicalGlobalObject, throwScope, IDLDouble::extractValueFromNullable(dictionary.member));
        RETURN_IF_EXCEPTION(throwScope, { });
        result->putDirect(vm, JSC::Identifier::fromString(vm, "member"_s), memberValue);
    }
    return result;
}

template<> ConversionResult<IDLDictionary<TestDictionaryNoToNative::GenerateKeyword>> convertDictionary<TestDictionaryNoToNative::GenerateKeyword>(JSGlobalObject& lexicalGlobalObject, JSValue value)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    bool isNullOrUndefined = value.isUndefinedOrNull();
    auto* object = isNullOrUndefined ? nullptr : value.getObject();
    if (UNLIKELY(!isNullOrUndefined && !object)) {
        throwTypeError(&lexicalGlobalObject, throwScope);
        return ConversionResultException { };
    }
    TestDictionaryNoToNative::GenerateKeyword result;
    JSValue memberValue;
    if (isNullOrUndefined)
        memberValue = jsUndefined();
    else {
        memberValue = object->get(&lexicalGlobalObject, Identifier::fromString(vm, "member"_s));
        RETURN_IF_EXCEPTION(throwScope, ConversionResultException { });
    }
    if (!memberValue.isUndefined()) {
        auto memberConversionResult = convert<IDLDouble>(lexicalGlobalObject, memberValue);
        if (UNLIKELY(memberConversionResult.hasException(throwScope)))
            return ConversionResultException { };
        result.member = memberConversionResult.releaseReturnValue();
    }
    return result;
}

} // namespace WebCore

