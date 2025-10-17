/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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

#include "SVGGraphicsElement.h"
#include "SVGURIReference.h"
#include "SharedStringHash.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DOMTokenList;

class SVGAElement final : public SVGGraphicsElement, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGAElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGAElement);
public:
    static Ref<SVGAElement> create(const QualifiedName&, Document&);
    ~SVGAElement();

    AtomString target() const final { return AtomString { m_target->currentValue() }; }
    Ref<SVGAnimatedString>& targetAnimated() { return m_target; }

    SharedStringHash visitedLinkHash() const;

    DOMTokenList& relList();

private:
    SVGAElement(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGAElement, SVGGraphicsElement, SVGURIReference>;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool childShouldCreateRenderer(const Node&) const final;

    bool isValid() const final { return SVGTests::isValid(); }
    String title() const final;
    void defaultEventHandler(Event&) final;
    
    bool supportsFocus() const final;
    bool isMouseFocusable() const final;
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isURLAttribute(const Attribute&) const final;
    bool canStartSelection() const final;
    int defaultTabIndex() const final;

    bool willRespondToMouseClickEventsWithEditability(Editability) const final;

    Ref<SVGAnimatedString> m_target { SVGAnimatedString::create(this) };

    // This is computed only once and must not be affected by subsequent URL changes.
    mutable std::optional<SharedStringHash> m_storedVisitedLinkHash;

    std::unique_ptr<DOMTokenList> m_relList;
};

} // namespace WebCore
