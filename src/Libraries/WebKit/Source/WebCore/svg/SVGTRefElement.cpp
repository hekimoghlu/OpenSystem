/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#include "SVGTRefElement.h"

#include "AddEventListenerOptions.h"
#include "ElementRareData.h"
#include "EventListener.h"
#include "EventNames.h"
#include "LegacyRenderSVGResource.h"
#include "MutationEvent.h"
#include "RenderSVGInline.h"
#include "RenderSVGInlineText.h"
#include "SVGDocumentExtensions.h"
#include "SVGElementInlines.h"
#include "SVGNames.h"
#include "ScriptDisallowedScope.h"
#include "ShadowRoot.h"
#include "StyleInheritedData.h"
#include "Text.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGTRefElement);

Ref<SVGTRefElement> SVGTRefElement::create(const QualifiedName& tagName, Document& document)
{
    Ref element = adoptRef(*new SVGTRefElement(tagName, document));
    element->ensureUserAgentShadowRoot();
    return element;
}

class SVGTRefTargetEventListener final : public EventListener {
public:
    static Ref<SVGTRefTargetEventListener> create(SVGTRefElement& trefElement)
    {
        return adoptRef(*new SVGTRefTargetEventListener(trefElement));
    }

    static const SVGTRefTargetEventListener* cast(const EventListener* listener)
    {
        return listener->type() == SVGTRefTargetEventListenerType ? static_cast<const SVGTRefTargetEventListener*>(listener) : nullptr;
    }

    void attach(RefPtr<Element>&& target);
    void detach();
    bool isAttached() const { return m_target.get(); }

private:
    explicit SVGTRefTargetEventListener(SVGTRefElement& trefElement);

    Ref<SVGTRefElement> protectedTRefElement() const { return m_trefElement.get(); }
    RefPtr<Element> protectedTarget() const { return m_target; }
    void handleEvent(ScriptExecutionContext&, Event&) final;
    bool operator==(const EventListener&) const final;

    WeakRef<SVGTRefElement, WeakPtrImplWithEventTargetData> m_trefElement;
    RefPtr<Element> m_target;
};

SVGTRefTargetEventListener::SVGTRefTargetEventListener(SVGTRefElement& trefElement)
    : EventListener(SVGTRefTargetEventListenerType)
    , m_trefElement(trefElement)
{
}

void SVGTRefTargetEventListener::attach(RefPtr<Element>&& target)
{
    ASSERT(!isAttached());
    ASSERT(target.get());
    ASSERT(target->isConnected());

    target->addEventListener(eventNames().DOMSubtreeModifiedEvent, *this, false);
    target->addEventListener(eventNames().DOMNodeRemovedFromDocumentEvent, *this, false);
    m_target = WTFMove(target);
}

void SVGTRefTargetEventListener::detach()
{
    if (!isAttached())
        return;

    RefPtr target = m_target;
    target->removeEventListener(eventNames().DOMSubtreeModifiedEvent, *this, false);
    target->removeEventListener(eventNames().DOMNodeRemovedFromDocumentEvent, *this, false);
    m_target = nullptr;
}

bool SVGTRefTargetEventListener::operator==(const EventListener& listener) const
{
    if (const SVGTRefTargetEventListener* targetListener = SVGTRefTargetEventListener::cast(&listener))
        return &m_trefElement == &targetListener->m_trefElement;
    return false;
}

void SVGTRefTargetEventListener::handleEvent(ScriptExecutionContext&, Event& event)
{
    if (!isAttached())
        return;

    if (event.type() == eventNames().DOMSubtreeModifiedEvent && m_trefElement.ptr() != event.target())
        protectedTRefElement()->updateReferencedText(protectedTarget().get());
    else if (event.type() == eventNames().DOMNodeRemovedFromDocumentEvent)
        protectedTRefElement()->detachTarget();
}

inline SVGTRefElement::SVGTRefElement(const QualifiedName& tagName, Document& document)
    : SVGTextPositioningElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGURIReference(this)
    , m_targetListener(SVGTRefTargetEventListener::create(*this))
{
    ASSERT(hasTagName(SVGNames::trefTag));
}

SVGTRefElement::~SVGTRefElement()
{
    protectedTargetListener()->detach();
}

Ref<SVGTRefTargetEventListener> SVGTRefElement::protectedTargetListener() const
{
    return m_targetListener;
}

void SVGTRefElement::updateReferencedText(Element* target)
{
    String textContent;
    if (target)
        textContent = target->textContent();

    RefPtr root = userAgentShadowRoot();
    ASSERT(root);
    ScriptDisallowedScope::EventAllowedScope allowedScope(*root);
    if (!root->firstChild())
        root->appendChild(Text::create(protectedDocument(), WTFMove(textContent)));
    else {
        ASSERT(root->firstChild()->isTextNode());
        root->protectedFirstChild()->setTextContent(WTFMove(textContent));
    }
}

void SVGTRefElement::detachTarget()
{
    // Remove active listeners and clear the text content.
    protectedTargetListener()->detach();

    ASSERT(shadowRoot());
    RefPtr container = shadowRoot()->firstChild();
    if (container)
        container->setTextContent(String { });

    if (!isConnected())
        return;

    // Mark the referenced ID as pending.
    auto target = SVGURIReference::targetElementFromIRIString(href(), protectedDocument());
    if (!target.identifier.isEmpty())
        treeScopeForSVGReferences().addPendingSVGResource(target.identifier, *this);
}

void SVGTRefElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGURIReference::parseAttribute(name, newValue);
    SVGTextPositioningElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGTRefElement::svgAttributeChanged(const QualifiedName& attrName)
{
    SVGTextPositioningElement::svgAttributeChanged(attrName);

    if (SVGURIReference::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        buildPendingResource();
        updateSVGRendererForElementChange();
    }
}

RenderPtr<RenderElement> SVGTRefElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderSVGInline>(RenderObject::Type::SVGInline, *this, WTFMove(style));
}

bool SVGTRefElement::childShouldCreateRenderer(const Node& child) const
{
    return child.isInShadowTree();
}

bool SVGTRefElement::rendererIsNeeded(const RenderStyle& style)
{
    if (parentNode()
        && (parentNode()->hasTagName(SVGNames::aTag)
            || parentNode()->hasTagName(SVGNames::altGlyphTag)
            || parentNode()->hasTagName(SVGNames::textTag)
            || parentNode()->hasTagName(SVGNames::textPathTag)
            || parentNode()->hasTagName(SVGNames::tspanTag)))
        return StyledElement::rendererIsNeeded(style);

    return false;
}

void SVGTRefElement::clearTarget()
{
    protectedTargetListener()->detach();
}

void SVGTRefElement::buildPendingResource()
{
    // Remove any existing event listener.
    protectedTargetListener()->detach();

    // If we're not yet in a document, this function will be called again from insertedIntoAncestor().
    if (!isConnected())
        return;

    auto target = SVGURIReference::targetElementFromIRIString(href(), treeScopeForSVGReferences());
    if (!target.element) {
        if (target.identifier.isEmpty())
            return;

        treeScopeForSVGReferences().addPendingSVGResource(target.identifier, *this);
        ASSERT(hasPendingResources());
        return;
    }

    // Don't set up event listeners if this is a shadow tree node.
    // SVGUseElement::transferEventListenersToShadowTree() handles this task, and addEventListener()
    // expects every element instance to have an associated shadow tree element - which is not the
    // case when we land here from SVGUseElement::buildShadowTree().
    if (!isInShadowTree())
        protectedTargetListener()->attach(target.element.copyRef());

    updateReferencedText(target.element.get());
}

Node::InsertedIntoAncestorResult SVGTRefElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    auto result = SVGElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
    return result;
}

void SVGTRefElement::didFinishInsertingNode()
{
    SVGTextPositioningElement::didFinishInsertingNode();
    buildPendingResource();
}

void SVGTRefElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    SVGElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (removalType.disconnectedFromDocument)
        protectedTargetListener()->detach();
}

}
