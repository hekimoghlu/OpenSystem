/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include "CustomElementRegistry.h"

#include "CustomElementReactionQueue.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "ElementRareData.h"
#include "ElementTraversal.h"
#include "JSCustomElementInterface.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalDOMWindow.h"
#include "MathMLNames.h"
#include "QualifiedName.h"
#include "Quirks.h"
#include "ShadowRoot.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

Ref<CustomElementRegistry> CustomElementRegistry::create(ScriptExecutionContext& scriptExecutionContext, LocalDOMWindow& window)
{
    return adoptRef(*new CustomElementRegistry(scriptExecutionContext, window));
}

Ref<CustomElementRegistry> CustomElementRegistry::create(ScriptExecutionContext& scriptExecutionContext)
{
    return adoptRef(*new CustomElementRegistry(scriptExecutionContext));
}

CustomElementRegistry::CustomElementRegistry(ScriptExecutionContext& scriptExecutionContext, LocalDOMWindow& window)
    : ContextDestructionObserver(&scriptExecutionContext)
    , m_window(window)
{
}

CustomElementRegistry::CustomElementRegistry(ScriptExecutionContext& scriptExecutionContext)
    : ContextDestructionObserver(&scriptExecutionContext)
{
}

CustomElementRegistry::~CustomElementRegistry() = default;

Document* CustomElementRegistry::document() const
{
    return m_window ? m_window->document() : nullptr;
}

void CustomElementRegistry::didAssociateWithDocument(Document& document)
{
    m_associatedDocuments.add(document);
}

// https://dom.spec.whatwg.org/#concept-shadow-including-tree-order
static void enqueueUpgradeInShadowIncludingTreeOrder(ContainerNode& node, JSCustomElementInterface& elementInterface, CustomElementRegistry& registry)
{
    for (RefPtr element = ElementTraversal::firstWithin(node); element; element = ElementTraversal::next(*element)) {
        if (element->isCustomElementUpgradeCandidate() && element->treeScope().customElementRegistry() == &registry && element->tagQName().matches(elementInterface.name()))
            element->enqueueToUpgrade(elementInterface);
        if (RefPtr shadowRoot = element->shadowRoot()) {
            if (shadowRoot->mode() != ShadowRootMode::UserAgent)
                enqueueUpgradeInShadowIncludingTreeOrder(*shadowRoot, elementInterface, registry);
        }
    }
}

RefPtr<DeferredPromise> CustomElementRegistry::addElementDefinition(Ref<JSCustomElementInterface>&& elementInterface)
{
    static MainThreadNeverDestroyed<const AtomString> extendsLi("extends-li"_s);

    AtomString localName = elementInterface->name().localName();
    ASSERT(!m_nameMap.contains(localName));
    m_nameMap.add(localName, elementInterface.copyRef());
    {
        Locker locker { m_constructorMapLock };
        m_constructorMap.add(elementInterface->constructor(), elementInterface.ptr());
    }

    if (elementInterface->isShadowDisabled())
        m_disabledShadowSet.add(localName);

    if (RefPtr document = this->document()) { // Global custom element registry
        // ungap/@custom-elements detection for quirk (rdar://problem/111008826).
        if (localName == extendsLi.get())
            document->quirks().setNeedsConfigurableIndexedPropertiesQuirk();
        enqueueUpgradeInShadowIncludingTreeOrder(*document, elementInterface.get(), *this);
    }

    for (Ref document : m_associatedDocuments) {
        if (document->hasBrowsingContext())
            enqueueUpgradeInShadowIncludingTreeOrder(document, elementInterface.get(), *this);
    }

    return m_promiseMap.take(localName);
}

JSCustomElementInterface* CustomElementRegistry::findInterface(const Element& element) const
{
    return findInterface(element.tagQName());
}

JSCustomElementInterface* CustomElementRegistry::findInterface(const QualifiedName& name) const
{
    if (name.namespaceURI() != HTMLNames::xhtmlNamespaceURI)
        return nullptr;
    return m_nameMap.get(name.localName());
}

JSCustomElementInterface* CustomElementRegistry::findInterface(const AtomString& name) const
{
    return m_nameMap.get(name);
}

RefPtr<JSCustomElementInterface> CustomElementRegistry::findInterface(const JSC::JSObject* constructor) const
{
    Locker locker { m_constructorMapLock };
    return m_constructorMap.get(constructor);
}

bool CustomElementRegistry::containsConstructor(const JSC::JSObject* constructor) const
{
    Locker locker { m_constructorMapLock };
    return m_constructorMap.contains(constructor);
}

JSC::JSValue CustomElementRegistry::get(const AtomString& name)
{
    if (RefPtr elementInterface = m_nameMap.get(name))
        return elementInterface->constructor();
    return JSC::jsUndefined();
}

String CustomElementRegistry::getName(JSC::JSValue constructorValue)
{
    auto* constructor = constructorValue.getObject();
    if (!constructor)
        return String { };
    RefPtr elementInterface = findInterface(constructor);
    if (!elementInterface)
        return String { };
    return elementInterface->name().localName();
}

static void upgradeElementsInShadowIncludingDescendants(ContainerNode& root)
{
    for (Ref element : descendantsOfType<Element>(root)) {
        if (element->isCustomElementUpgradeCandidate())
            CustomElementReactionQueue::tryToUpgradeElement(element);
        if (RefPtr shadowRoot = element->shadowRoot())
            upgradeElementsInShadowIncludingDescendants(*shadowRoot);
    }
}

void CustomElementRegistry::upgrade(Node& root)
{
    auto* containerNode = dynamicDowncast<ContainerNode>(root);
    if (!containerNode)
        return;

    RefPtr element = dynamicDowncast<Element>(*containerNode);
    if (element && element->isCustomElementUpgradeCandidate())
        CustomElementReactionQueue::tryToUpgradeElement(*element);

    upgradeElementsInShadowIncludingDescendants(*containerNode);
}

void CustomElementRegistry::addToScopedCustomElementRegistryMap(Element& element, CustomElementRegistry& registry)
{
    ASSERT(!element.usesScopedCustomElementRegistryMap());
    element.setUsesScopedCustomElementRegistryMap();
    auto result = scopedCustomElementRegistryMap().add(element, registry);
    ASSERT_UNUSED(result, result.isNewEntry);
}

void CustomElementRegistry::removeFromScopedCustomElementRegistryMap(Element& element)
{
    ASSERT(element.usesScopedCustomElementRegistryMap());
    element.clearUsesScopedCustomElementRegistryMap();
    auto didRemove = scopedCustomElementRegistryMap().remove(element);
    ASSERT_UNUSED(didRemove, didRemove);
}

WeakHashMap<Element, Ref<CustomElementRegistry>, WeakPtrImplWithEventTargetData>& CustomElementRegistry::scopedCustomElementRegistryMap()
{
    static NeverDestroyed<WeakHashMap<Element, Ref<CustomElementRegistry>, WeakPtrImplWithEventTargetData>> map;
    return map.get();
}

template<typename Visitor>
void CustomElementRegistry::visitJSCustomElementInterfaces(Visitor& visitor) const
{
    Locker locker { m_constructorMapLock };
    for (const auto& iterator : m_constructorMap)
        iterator.value->visitJSFunctions(visitor);
}

template void CustomElementRegistry::visitJSCustomElementInterfaces(JSC::AbstractSlotVisitor&) const;
template void CustomElementRegistry::visitJSCustomElementInterfaces(JSC::SlotVisitor&) const;

}
