/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include "JSElement.h"

#include "Document.h"
#include "HTMLFrameElementBase.h"
#include "HTMLNames.h"
#include "JSAttr.h"
#include "JSDOMBinding.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertNullable.h"
#include "JSDOMConvertSequences.h"
#include "JSHTMLElementWrapperFactory.h"
#include "JSMathMLElementWrapperFactory.h"
#include "JSNodeList.h"
#include "JSSVGElementWrapperFactory.h"
#include "MathMLElement.h"
#include "NodeList.h"
#include "SVGElement.h"
#include "WebCoreJSClientData.h"


namespace WebCore {
using namespace JSC;

using namespace HTMLNames;

static JSValue createNewElementWrapper(JSDOMGlobalObject* globalObject, Ref<Element>&& element)
{
    if (auto* htmlElement = dynamicDowncast<HTMLElement>(element.get()))
        return createJSHTMLWrapper(globalObject, *htmlElement);
    if (auto* svgElement = dynamicDowncast<SVGElement>(element.get()))
        return createJSSVGWrapper(globalObject, *svgElement);
#if ENABLE(MATHML)
    if (auto* mathmlElement = dynamicDowncast<MathMLElement>(element.get()))
        return createJSMathMLWrapper(globalObject, *mathmlElement);
#endif
    return createWrapper<Element>(globalObject, WTFMove(element));
}

JSValue toJS(JSGlobalObject*, JSDOMGlobalObject* globalObject, Element& element)
{
    if (auto* wrapper = getCachedWrapper(globalObject->world(), element))
        return wrapper;
    return createNewElementWrapper(globalObject, element);
}

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<Element>&& element)
{
    if (element->isDefinedCustomElement()) {
        JSValue result = getCachedWrapper(globalObject->world(), element);
        if (result)
            return result;
        ASSERT(!globalObject->vm().exceptionForInspection());
    }
    ASSERT(!getCachedWrapper(globalObject->world(), element));
    return createNewElementWrapper(globalObject, WTFMove(element));
}

static JSValue getElementsArrayAttribute(JSGlobalObject& lexicalGlobalObject, const JSElement& thisObject, const QualifiedName& attributeName)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSObject* cachedObject = nullptr;
    JSValue cachedObjectValue = thisObject.getDirect(vm, builtinNames(vm).cachedAttrAssociatedElementsPrivateName());
    if (cachedObjectValue)
        cachedObject = asObject(cachedObjectValue);
    else {
        cachedObject = constructEmptyObject(vm, thisObject.globalObject()->nullPrototypeObjectStructure());
        const_cast<JSElement&>(thisObject).putDirect(vm, builtinNames(vm).cachedAttrAssociatedElementsPrivateName(), cachedObject);
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

JSValue JSElement::ariaControlsElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_controlsAttr);
}

JSValue JSElement::ariaDescribedByElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_describedbyAttr);
}

JSValue JSElement::ariaDetailsElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_detailsAttr);
}

JSValue JSElement::ariaErrorMessageElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_errormessageAttr);
}

JSValue JSElement::ariaFlowToElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_flowtoAttr);
}

JSValue JSElement::ariaLabelledByElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_labelledbyAttr);
}

JSValue JSElement::ariaOwnsElements(JSGlobalObject& lexicalGlobalObject) const
{
    return getElementsArrayAttribute(lexicalGlobalObject, *this, WebCore::HTMLNames::aria_ownsAttr);
}

} // namespace WebCore
