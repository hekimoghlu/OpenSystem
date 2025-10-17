/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include "JSHTMLElement.h"

#include "CustomElementRegistry.h"
#include "Document.h"
#include "FormAssociatedElement.h"
#include "HTMLFormControlElement.h"
#include "HTMLFormElement.h"
#include "JSCustomElementInterface.h"
#include "JSDOMConstructorBase.h"
#include "JSHTMLElementWrapperFactory.h"
#include "JSNodeCustom.h"
#include "LocalDOMWindow.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/InternalFunction.h>
#include <JavaScriptCore/JSWithScope.h>

namespace WebCore {

using namespace JSC;

EncodedJSValue constructJSHTMLElement(JSGlobalObject* lexicalGlobalObject, CallFrame& callFrame)
{
    VM& vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* jsConstructor = jsCast<JSDOMConstructorBase*>(callFrame.jsCallee());
    ASSERT(jsConstructor);

    auto* context = jsConstructor->scriptExecutionContext();
    if (!context)
        return throwConstructorScriptExecutionContextUnavailableError(*lexicalGlobalObject, scope, "HTMLElement"_s);
    ASSERT(context->isDocument());

    auto* newTarget = callFrame.newTarget().getObject();
    auto* functionGlobalObject = getFunctionRealm(lexicalGlobalObject, newTarget);
    RETURN_IF_EXCEPTION(scope, { });
    auto* newTargetGlobalObject = jsCast<JSDOMGlobalObject*>(functionGlobalObject);
    JSValue htmlElementConstructorValue = JSHTMLElement::getConstructor(vm, newTargetGlobalObject);
    if (newTarget == htmlElementConstructorValue)
        return throwVMTypeError(lexicalGlobalObject, scope, "new.target is not a valid custom element constructor"_s);

    Ref document = downcast<Document>(*context);

    RefPtr registry = document->activeCustomElementRegistry();
    if (!registry) {
        RefPtr window = document->domWindow();
        if (!window)
            return throwVMTypeError(lexicalGlobalObject, scope, "new.target is not a valid custom element constructor"_s);

        registry = window->customElementRegistry();
        if (!registry)
            return throwVMTypeError(lexicalGlobalObject, scope, "new.target is not a valid custom element constructor"_s);
    }

    RefPtr elementInterface = registry->findInterface(newTarget);
    if (!elementInterface)
        return throwVMTypeError(lexicalGlobalObject, scope, "new.target does not define a custom element"_s);

    if (!elementInterface->isUpgradingElement()) {
        Structure* baseStructure = getDOMStructure<JSHTMLElement>(vm, *newTargetGlobalObject);
        auto* newElementStructure = InternalFunction::createSubclassStructure(lexicalGlobalObject, newTarget, baseStructure);
        RETURN_IF_EXCEPTION(scope, { });

        Ref element = elementInterface->createElement(document);
        if (registry->isScoped())
            CustomElementRegistry::addToScopedCustomElementRegistryMap(element, *registry);
        element->setIsDefinedCustomElement(*elementInterface);
        auto* jsElement = JSHTMLElement::create(newElementStructure, newTargetGlobalObject, element.get());
        cacheWrapper(newTargetGlobalObject->world(), element.ptr(), jsElement);
        return JSValue::encode(jsElement);
    }

    RefPtr elementToUpgrade = elementInterface->lastElementInConstructionStack();
    if (!elementToUpgrade) {
        throwTypeError(lexicalGlobalObject, scope, "Cannot instantiate a custom element inside its own constructor during upgrades"_s);
        return JSValue::encode(jsUndefined());
    }

    JSValue elementWrapperValue = toJS(lexicalGlobalObject, jsConstructor->globalObject(), *elementToUpgrade);
    ASSERT(elementWrapperValue.isObject());

    JSValue newPrototype = newTarget->get(lexicalGlobalObject, vm.propertyNames->prototype);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    JSObject* elementWrapperObject = asObject(elementWrapperValue);
    JSObject::setPrototype(elementWrapperObject, lexicalGlobalObject, newPrototype, true /* shouldThrowIfCantSet */);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    elementInterface->didUpgradeLastElementInConstructionStack();

    return JSValue::encode(elementWrapperValue);
}

JSScope* JSHTMLElement::pushEventHandlerScope(JSGlobalObject* lexicalGlobalObject, JSScope* scope) const
{
    HTMLElement& element = wrapped();

    // The document is put on first, fall back to searching it only after the element and form.
    // FIXME: This probably may use the wrong global object. If this is called from a native
    // function, then it would be correct but not optimal since the native function would *know*
    // the global object. But, it may be that globalObject() is more correct.
    // https://bugs.webkit.org/show_bug.cgi?id=134932
    VM& vm = lexicalGlobalObject->vm();
    
    scope = JSWithScope::create(vm, lexicalGlobalObject, scope, asObject(toJS(lexicalGlobalObject, globalObject(), element.document())));

    // The form is next, searched before the document, but after the element itself.
    if (auto* formAssociated = element.asFormAssociatedElement()) {
        if (RefPtr form = formAssociated->form())
            scope = JSWithScope::create(vm, lexicalGlobalObject, scope, asObject(toJS(lexicalGlobalObject, globalObject(), *form)));
    }

    // The element is on top, searched first.
    return JSWithScope::create(vm, lexicalGlobalObject, scope, asObject(toJS(lexicalGlobalObject, globalObject(), element)));
}

JSValue toJS(JSGlobalObject*, JSDOMGlobalObject* globalObject, HTMLElement& element)
{
    if (auto* wrapper = getCachedWrapper(globalObject->world(), element))
        return wrapper;
    return createJSHTMLWrapper(globalObject, element);
}

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<HTMLElement>&& element)
{
    if (element->isDefinedCustomElement()) {
        JSValue result = getCachedWrapper(globalObject->world(), element);
        if (result)
            return result;
        ASSERT(!globalObject->vm().exceptionForInspection());
    }
    ASSERT(!getCachedWrapper(globalObject->world(), element));
    return createJSHTMLWrapper(globalObject, WTFMove(element));
}

} // namespace WebCore
