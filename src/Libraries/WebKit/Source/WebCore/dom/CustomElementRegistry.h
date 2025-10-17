/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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

#include "ContextDestructionObserver.h"
#include "Element.h"
#include "EventTarget.h"
#include "QualifiedName.h"
#include "TreeScope.h"
#include <wtf/Lock.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/WeakHashMap.h>
#include <wtf/WeakListHashSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace JSC {

class JSGlobalObject;
class JSObject;
class JSValue;

}

namespace WebCore {

class CustomElementRegistry;
class LocalDOMWindow;
class DeferredPromise;
class Document;
class Element;
class JSCustomElementInterface;
class Node;
class QualifiedName;

class CustomElementRegistry : public RefCounted<CustomElementRegistry>, public ContextDestructionObserver {
public:
    static Ref<CustomElementRegistry> create(ScriptExecutionContext&, LocalDOMWindow&);
    static Ref<CustomElementRegistry> create(ScriptExecutionContext&);
    ~CustomElementRegistry();

    bool isScoped() const { return !m_window; }
    Document* document() const;

    static CustomElementRegistry* registryForElement(const Element& element)
    {
        if (UNLIKELY(element.usesScopedCustomElementRegistryMap()))
            return scopedCustomElementRegistryMap().get(element);
        return element.treeScope().customElementRegistry();
    }

    static CustomElementRegistry* registryForNodeOrTreeScope(const Node& node, const TreeScope& treeScope)
    {
        if (auto* element = dynamicDowncast<Element>(node); UNLIKELY(element && element->usesScopedCustomElementRegistryMap()))
            return scopedCustomElementRegistryMap().get(*element);
        return treeScope.customElementRegistry();
    }

    static void addToScopedCustomElementRegistryMap(Element&, CustomElementRegistry&);
    static void removeFromScopedCustomElementRegistryMap(Element&);

    void didAssociateWithDocument(Document&);

    RefPtr<DeferredPromise> addElementDefinition(Ref<JSCustomElementInterface>&&);

    bool& elementDefinitionIsRunning() { return m_elementDefinitionIsRunning; }

    JSCustomElementInterface* findInterface(const Element&) const;
    JSCustomElementInterface* findInterface(const QualifiedName&) const;
    JSCustomElementInterface* findInterface(const AtomString&) const;
    RefPtr<JSCustomElementInterface> findInterface(const JSC::JSObject*) const;
    bool containsConstructor(const JSC::JSObject*) const;

    JSC::JSValue get(const AtomString&);
    String getName(JSC::JSValue);
    void upgrade(Node& root);

    MemoryCompactRobinHoodHashMap<AtomString, Ref<DeferredPromise>>& promiseMap() { return m_promiseMap; }
    bool isShadowDisabled(const AtomString& name) const { return m_disabledShadowSet.contains(name); }

    template<typename Visitor> void visitJSCustomElementInterfaces(Visitor&) const;

private:
    CustomElementRegistry(ScriptExecutionContext&, LocalDOMWindow&);
    CustomElementRegistry(ScriptExecutionContext&);

    static WeakHashMap<Element, Ref<CustomElementRegistry>, WeakPtrImplWithEventTargetData>& scopedCustomElementRegistryMap();

    WeakPtr<LocalDOMWindow, WeakPtrImplWithEventTargetData> m_window;
    UncheckedKeyHashMap<AtomString, Ref<JSCustomElementInterface>> m_nameMap;
    UncheckedKeyHashMap<const JSC::JSObject*, JSCustomElementInterface*> m_constructorMap WTF_GUARDED_BY_LOCK(m_constructorMapLock);
    MemoryCompactRobinHoodHashMap<AtomString, Ref<DeferredPromise>> m_promiseMap;
    MemoryCompactRobinHoodHashSet<AtomString> m_disabledShadowSet;
    WeakListHashSet<Document, WeakPtrImplWithEventTargetData> m_associatedDocuments;

    bool m_elementDefinitionIsRunning { false };
    mutable Lock m_constructorMapLock;

    friend class ElementDefinitionIsRunningSetForScope;
};

}
