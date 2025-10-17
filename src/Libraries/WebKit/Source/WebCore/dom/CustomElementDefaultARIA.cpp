/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "CustomElementDefaultARIA.h"

#include "Element.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include "SpaceSplitString.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CustomElementDefaultARIA);

CustomElementDefaultARIA::CustomElementDefaultARIA() = default;
CustomElementDefaultARIA::~CustomElementDefaultARIA() = default;

void CustomElementDefaultARIA::setValueForAttribute(const QualifiedName& name, const AtomString& value)
{
    m_map.set(name, value);
}

static bool isElementVisible(const Element& element, const Element& thisElement)
{
    return !element.isConnected() || element.isInDocumentTree() || thisElement.isDescendantOrShadowDescendantOf(element.protectedRootNode());
}

const AtomString& CustomElementDefaultARIA::valueForAttribute(const Element& thisElement, const QualifiedName& name) const
{
    auto it = m_map.find(name);
    if (it == m_map.end())
        return nullAtom();

    return std::visit(WTF::makeVisitor([&](const AtomString& stringValue) -> const AtomString& {
        return stringValue;
    }, [&](const WeakPtr<Element, WeakPtrImplWithEventTargetData>& weakElementValue) -> const AtomString& {
        RefPtr elementValue = weakElementValue.get();
        if (elementValue && isElementVisible(*elementValue, thisElement))
            return elementValue->attributeWithoutSynchronization(HTMLNames::idAttr);
        return nullAtom();
    }, [&](const Vector<WeakPtr<Element, WeakPtrImplWithEventTargetData>>& elements) -> const AtomString& {
        StringBuilder idList;
        for (auto& weakElement : elements) {
            RefPtr element = weakElement.get();
            if (element && isElementVisible(*element, thisElement)) {
                if (idList.length())
                    idList.append(' ');
                idList.append(element->attributeWithoutSynchronization(HTMLNames::idAttr));
            }
        }
        // FIXME: This should probably be using the idList we just built.
        return nullAtom();
    }), it->value);
}

bool CustomElementDefaultARIA::hasAttribute(const QualifiedName& name) const
{
    return m_map.find(name) != m_map.end();
}

RefPtr<Element> CustomElementDefaultARIA::elementForAttribute(const Element& thisElement, const QualifiedName& name) const
{
    auto it = m_map.find(name);
    if (it == m_map.end())
        return nullptr;

    RefPtr<Element> result;
    std::visit(WTF::makeVisitor([&](const AtomString& stringValue) {
        if (thisElement.isInTreeScope())
            result = thisElement.treeScope().getElementById(stringValue);
    }, [&](const WeakPtr<Element, WeakPtrImplWithEventTargetData>& weakElementValue) {
        RefPtr elementValue = weakElementValue.get();
        if (elementValue && isElementVisible(*elementValue, thisElement))
            result = WTFMove(elementValue);
    }, [&](const Vector<WeakPtr<Element, WeakPtrImplWithEventTargetData>>&) {
        RELEASE_ASSERT_NOT_REACHED();
    }), it->value);
    return result;
}

void CustomElementDefaultARIA::setElementForAttribute(const QualifiedName& name, Element* element)
{
    m_map.set(name, WeakPtr<Element, WeakPtrImplWithEventTargetData> { element });
}

Vector<Ref<Element>> CustomElementDefaultARIA::elementsForAttribute(const Element& thisElement, const QualifiedName& name) const
{
    Vector<Ref<Element>> result;
    auto it = m_map.find(name);
    if (it == m_map.end())
        return result;
    std::visit(WTF::makeVisitor([&](const AtomString& stringValue) {
        if (thisElement.isInTreeScope()) {
            SpaceSplitString idList { stringValue, SpaceSplitString::ShouldFoldCase::No };
            result = WTF::compactMap(idList, [&](auto& id) {
                return thisElement.treeScope().getElementById(id);
            });
        }
    }, [&](const WeakPtr<Element, WeakPtrImplWithEventTargetData>& weakElementValue) {
        RefPtr element = weakElementValue.get();
        if (element && isElementVisible(*element, thisElement))
            result.append(element.releaseNonNull());
    }, [&](const Vector<WeakPtr<Element, WeakPtrImplWithEventTargetData>>& elements) {
        result.reserveInitialCapacity(elements.size());
        for (auto& weakElement : elements) {
            if (RefPtr element = weakElement.get(); element && isElementVisible(*element, thisElement))
                result.append(element.releaseNonNull());
        }
    }), it->value);
    return result;
}

void CustomElementDefaultARIA::setElementsForAttribute(const QualifiedName& name, std::optional<Vector<Ref<Element>>>&& values)
{
    Vector<WeakPtr<Element, WeakPtrImplWithEventTargetData>> elements;
    if (values) {
        for (auto& element : *values) {
            elements.append(WeakPtr<Element, WeakPtrImplWithEventTargetData> { element });
        }
    }
    m_map.set(name, WTFMove(elements));
}

} // namespace WebCore
