/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#include "JSElementInternals.h"

#include "HTMLNames.h"
#include "IDLTypes.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertNullable.h"
#include "JSDOMConvertSequences.h"
#include "JSDOMConvertUnion.h"
#include "JSDOMFormData.h"
#include "JSElement.h"
#include "JSFile.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/ObjectConstructor.h>

namespace WebCore {
using namespace JSC;

JSValue JSElementInternals::setFormValue(JSGlobalObject& lexicalGlobalObject, CallFrame& callFrame)
{
    using JSCustomElementFormValue = IDLUnion<IDLNull, IDLInterface<File>, IDLUSVString, IDLInterface<DOMFormData>>;

    auto& vm = lexicalGlobalObject.vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    if (UNLIKELY(callFrame.argumentCount() < 1)) {
        throwException(&lexicalGlobalObject, throwScope, createNotEnoughArgumentsError(&lexicalGlobalObject));
        return { };
    }

    EnsureStillAliveScope argument0 = callFrame.uncheckedArgument(0);
    auto value = convert<JSCustomElementFormValue>(lexicalGlobalObject, argument0.value());
    if (UNLIKELY(value.hasException(throwScope)))
        return { };

    std::optional<CustomElementFormValue> state;
    if (callFrame.argumentCount() > 1) {
        EnsureStillAliveScope argument1 = callFrame.argument(1);
        auto stateConversionResult = convert<JSCustomElementFormValue>(lexicalGlobalObject, argument1.value());
        if (UNLIKELY(stateConversionResult.hasException(throwScope)))
            return { };
        state = stateConversionResult.releaseReturnValue();
    }

    auto result = wrapped().setFormValue(value.releaseReturnValue(), WTFMove(state));
    if (UNLIKELY(result.hasException())) {
        propagateException(lexicalGlobalObject, throwScope, result.releaseException());
        return { };
    }

    return jsUndefined();
}

static JSValue getElementsArrayAttribute(JSGlobalObject& lexicalGlobalObject, const JSElementInternals& thisObject, const QualifiedName& attributeName)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSObject* cachedObject = nullptr;
    JSValue cachedObjectValue = thisObject.getDirect(vm, builtinNames(vm).cachedAttrAssociatedElementsPrivateName());
    if (cachedObjectValue)
        cachedObject = asObject(cachedObjectValue);
    else {
        cachedObject = constructEmptyObject(vm, thisObject.globalObject()->nullPrototypeObjectStructure());
        const_cast<JSElementInternals&>(thisObject).putDirect(vm, builtinNames(vm).cachedAttrAssociatedElementsPrivateName(), cachedObject);
    }

    std::optional<Vector<Ref<Element>>> elements = thisObject.wrapped().getElementsArrayAttribute(attributeName);
    auto propertyName = PropertyName(Identifier::fromString(vm, attributeName.toString()));
    JSValue cachedValue = cachedObject->getDirect(vm, propertyName);
    if (!cachedValue.isEmpty()) {
        auto cachedElements = convert<IDLNullable<IDLFrozenArray<IDLInterface<Element>>>>(lexicalGlobalObject, cachedValue);
        if (!cachedElements.hasException(throwScope) && elements == cachedElements.returnValue())
            return cachedValue;
    }

    JSValue elementsValue = toJS<IDLNullable<IDLFrozenArray<IDLInterface<Element>>>>(lexicalGlobalObject, *thisObject.globalObject(), throwScope, elements);
    cachedObject->putDirect(vm, propertyName, elementsValue);
    return elementsValue;
}

JSValue JSElementInternals::ariaControlsElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_controlsAttr);
}

JSValue JSElementInternals::ariaDescribedByElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_describedbyAttr);
}

JSValue JSElementInternals::ariaDetailsElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_detailsAttr);
}

JSValue JSElementInternals::ariaErrorMessageElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_errormessageAttr);
}

JSValue JSElementInternals::ariaFlowToElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_flowtoAttr);
}

JSValue JSElementInternals::ariaLabelledByElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_labelledbyAttr);
}

JSValue JSElementInternals::ariaOwnsElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_ownsAttr);
}

} // namespace WebCore
