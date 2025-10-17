/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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

#include "ActiveDOMCallback.h"
#include "CustomElementFormValue.h"
#include "QualifiedName.h"
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Weak.h>
#include <JavaScriptCore/WeakInlines.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/RefCounted.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/text/AtomStringHash.h>

namespace JSC {
class JSObject;
class PrivateName;
}

namespace WebCore {

class CustomElementRegistry;
class DOMWrapperWorld;
class Document;
class Element;
class HTMLElement;
class JSDOMGlobalObject;

enum class ParserConstructElementWithEmptyStack : bool { No, Yes };

class JSCustomElementInterface : public RefCounted<JSCustomElementInterface>, public ActiveDOMCallback {
public:
    static Ref<JSCustomElementInterface> create(const QualifiedName& name, JSC::JSObject* callback, JSDOMGlobalObject* globalObject)
    {
        return adoptRef(*new JSCustomElementInterface(name, callback, globalObject));
    }

    Ref<Element> constructElementWithFallback(Document&, CustomElementRegistry&, const AtomString&, ParserConstructElementWithEmptyStack = ParserConstructElementWithEmptyStack::No);
    Ref<Element> constructElementWithFallback(Document&, CustomElementRegistry&, const QualifiedName&);
    Ref<HTMLElement> createElement(Document&);

    void upgradeElement(Element&);

    void setConnectedCallback(JSC::JSObject*);
    bool hasConnectedCallback() const { return !!m_connectedCallback; }
    void invokeConnectedCallback(Element&);

    void setDisconnectedCallback(JSC::JSObject*);
    bool hasDisconnectedCallback() const { return !!m_disconnectedCallback; }
    void invokeDisconnectedCallback(Element&);

    void setAdoptedCallback(JSC::JSObject*);
    bool hasAdoptedCallback() const { return !!m_adoptedCallback; }
    void invokeAdoptedCallback(Element&, Document& oldDocument, Document& newDocument);

    void setAttributeChangedCallback(JSC::JSObject* callback, Vector<AtomString>&& observedAttributes);
    bool observesAttribute(const AtomString& name) const { return m_observedAttributes.contains(name); }
    void invokeAttributeChangedCallback(Element&, const QualifiedName&, const AtomString& oldValue, const AtomString& newValue);

    void disableElementInternals() { m_isElementInternalsDisabled = true; }
    bool isElementInternalsDisabled() const { return m_isElementInternalsDisabled; }

    void disableShadow() { m_isShadowDisabled = true; }
    bool isShadowDisabled() const { return m_isShadowDisabled; }

    void setIsFormAssociated() { m_isFormAssociated = true; }
    bool isFormAssociated() const { return m_isFormAssociated; }

    void setFormAssociatedCallback(JSC::JSObject*);
    bool hasFormAssociatedCallback() const { return !!m_formAssociatedCallback; }
    void invokeFormAssociatedCallback(Element&, HTMLFormElement*);

    void setFormResetCallback(JSC::JSObject*);
    bool hasFormResetCallback() const { return !!m_formResetCallback; }
    void invokeFormResetCallback(Element&);

    void setFormDisabledCallback(JSC::JSObject*);
    bool hasFormDisabledCallback() const { return !!m_formDisabledCallback; }
    void invokeFormDisabledCallback(Element&, bool isDisabled);

    void setFormStateRestoreCallback(JSC::JSObject*);
    bool hasFormStateRestoreCallback() const { return !!m_formStateRestoreCallback; }
    void invokeFormStateRestoreCallback(Element&, CustomElementFormValue state);

    ScriptExecutionContext* scriptExecutionContext() const { return ContextDestructionObserver::scriptExecutionContext(); }
    JSC::JSObject* constructor() { return m_constructor.get(); }

    const QualifiedName& name() const { return m_name; }

    bool isUpgradingElement() const { return !m_constructionStack.isEmpty(); }
    Element* lastElementInConstructionStack() const { return m_constructionStack.last().get(); }
    void didUpgradeLastElementInConstructionStack();

    virtual ~JSCustomElementInterface();

    template<typename Visitor> void visitJSFunctions(Visitor&) const;
private:
    JSCustomElementInterface(const QualifiedName&, JSC::JSObject* callback, JSDOMGlobalObject*);

    RefPtr<Element> tryToConstructCustomElement(Document&, CustomElementRegistry&, const AtomString&, ParserConstructElementWithEmptyStack);

    template<typename Function>
    void invokeCallback(Element&, JSC::JSObject* callback, const Function& addArguments);

    QualifiedName m_name;
    JSC::Weak<JSC::JSObject> m_constructor;
    JSC::Weak<JSC::JSObject> m_connectedCallback;
    JSC::Weak<JSC::JSObject> m_disconnectedCallback;
    JSC::Weak<JSC::JSObject> m_adoptedCallback;
    JSC::Weak<JSC::JSObject> m_attributeChangedCallback;
    JSC::Weak<JSC::JSObject> m_formAssociatedCallback;
    JSC::Weak<JSC::JSObject> m_formResetCallback;
    JSC::Weak<JSC::JSObject> m_formDisabledCallback;
    JSC::Weak<JSC::JSObject> m_formStateRestoreCallback;
    Ref<DOMWrapperWorld> m_isolatedWorld;
    Vector<RefPtr<Element>, 1> m_constructionStack;
    MemoryCompactRobinHoodHashSet<AtomString> m_observedAttributes;
    bool m_isElementInternalsDisabled : 1;
    bool m_isShadowDisabled : 1;
    bool m_isFormAssociated : 1;
};

} // namespace WebCore
