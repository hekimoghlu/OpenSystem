/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

#include "SVGElement.h"
#include "SVGURIReference.h"
#include "ScriptElement.h"
#include "XLinkNames.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGScriptElement final : public SVGElement, public SVGURIReference, public ScriptElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGScriptElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGScriptElement);
public:
    static Ref<SVGScriptElement> create(const QualifiedName&, Document&, bool wasInsertedByParser);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGScriptElement, SVGElement, SVGURIReference>;
    using SVGElement::ref;
    using SVGElement::deref;

private:
    SVGScriptElement(const QualifiedName&, Document&, bool wasInsertedByParser, bool alreadyStarted);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void didFinishInsertingNode() final;
    void childrenChanged(const ChildChange&) final;
    void finishParsingChildren() final;

    bool isURLAttribute(const Attribute& attribute) const final { return attribute.name() == AtomString { sourceAttributeValue() }; }
    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;

    Ref<Element> cloneElementWithoutAttributesAndChildren(TreeScope&) final;
    bool rendererIsNeeded(const RenderStyle&) final { return false; }
    bool supportsFocus() const final { return false; }

    // ScriptElement
    String sourceAttributeValue() const final { return href(); }
    AtomString charsetAttributeValue() const final { return nullAtom(); }
    String typeAttributeValue() const final { return getAttribute(SVGNames::typeAttr).string(); }
    String languageAttributeValue() const final { return String(); }
    bool hasAsyncAttribute() const final { return false; }
    bool hasDeferAttribute() const final { return false; }
    bool hasNoModuleAttribute() const final { return false; }
    ReferrerPolicy referrerPolicy() const final { return ReferrerPolicy::EmptyString; }
    bool hasSourceAttribute() const final { return hasAttribute(SVGNames::hrefAttr) || hasAttribute(XLinkNames::hrefAttr); }
    void dispatchLoadEvent() final { SVGURIReference::dispatchLoadEvent(); }
    void dispatchErrorEvent() final;

    // SVGElement
    bool haveLoadedRequiredResources() final { return SVGURIReference::haveLoadedRequiredResources(); }
    Timer* loadEventTimer() final { return &m_loadEventTimer; }

    // SVGURIReference
    bool haveFiredLoadEvent() const final { return ScriptElement::haveFiredLoadEvent(); }
    void setHaveFiredLoadEvent(bool haveFiredLoadEvent) final { ScriptElement::setHaveFiredLoadEvent(haveFiredLoadEvent); }
    bool errorOccurred() const final { return ScriptElement::errorOccurred(); }
    void setErrorOccurred(bool errorOccurred) final { ScriptElement::setErrorOccurred(errorOccurred); }

#ifndef NDEBUG
    bool filterOutAnimatableAttribute(const QualifiedName& name) const final { return name == SVGNames::typeAttr; }
#endif

    Timer m_loadEventTimer;
};

} // namespace WebCore
