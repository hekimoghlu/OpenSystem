/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
#include "JSTestDictionaryWithOnlyConditionalMembers.h"

#include "JSDOMGlobalObject.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/ObjectConstructor.h>

#if ENABLE(TEST_CONDITIONAL)
#include "JSTestDictionary.h"
#endif



namespace WebCore {
using namespace JSC;

template<> ConversionResult<IDLDictionary<TestDictionaryWithOnlyConditionalMembers>> convertDictionary<TestDictionaryWithOnlyConditionalMembers>(JSGlobalObject& lexicalGlobalObject, JSValue value)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    bool isNullOrUndefined = value.isUndefinedOrNull();
    auto* object = isNullOrUndefined ? nullptr : value.getObject();
    if (UNLIKELY(!isNullOrUndefined && !object)) {
        throwTypeError(&lexicalGlobalObject, throwScope);
        return ConversionResultException { };
    }
    TestDictionaryWithOnlyConditionalMembers result;
#if ENABLE(TEST_CONDITIONAL)
    JSValue conditionalMemberValue;
    if (isNullOrUndefined)
        conditionalMemberValue = jsUndefined();
    else {
        conditionalMemberValue = object->get(&lexicalGlobalObject, Identifier::fromString(vm, "conditionalMember"_s));
        RETURN_IF_EXCEPTION(throwScope, ConversionResultException { });
    }
    if (!conditionalMemberValue.isUndefined()) {
        auto conditionalMemberConversionResult = convert<IDLDictionary<TestDictionary>>(lexicalGlobalObject, conditionalMemberValue);
        if (UNLIKELY(conditionalMemberConversionResult.hasException(throwScope)))
            return ConversionResultException { };
        result.conditionalMember = conditionalMemberConversionResult.releaseReturnValue();
    }
#endif
    return result;
}

JSC::JSObject* convertDictionaryToJS(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const TestDictionaryWithOnlyConditionalMembers& dictionary)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    auto result = constructEmptyObject(&lexicalGlobalObject, globalObject.objectPrototype());

#if ENABLE(TEST_CONDITIONAL)
    if (!IDLDictionary<TestDictionary>::isNullValue(dictionary.conditionalMember)) {
        auto conditionalMemberValue = toJS<IDLDictionary<TestDictionary>>(lexicalGlobalObject, globalObject, throwScope, IDLDictionary<TestDictionary>::extractValueFromNullable(dictionary.conditionalMember));
        RETURN_IF_EXCEPTION(throwScope, { });
        result->putDirect(vm, JSC::Identifier::fromString(vm, "conditionalMember"_s), conditionalMemberValue);
    }
#endif
    UNUSED_PARAM(dictionary);
    UNUSED_VARIABLE(throwScope);

    return result;
}

} // namespace WebCore

