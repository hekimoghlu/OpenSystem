/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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

#include "SVGTextPositioningElement.h"
#include "SVGURIReference.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGTRefTargetEventListener;

class SVGTRefElement final : public SVGTextPositioningElement, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGTRefElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGTRefElement);
public:
    static Ref<SVGTRefElement> create(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGTRefElement, SVGTextPositioningElement, SVGURIReference>;

private:
    friend class SVGTRefTargetEventListener;

    SVGTRefElement(const QualifiedName&, Document&);
    virtual ~SVGTRefElement();

    Ref<SVGTRefTargetEventListener> protectedTargetListener() const;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool childShouldCreateRenderer(const Node&) const override;
    bool rendererIsNeeded(const RenderStyle&) override;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) override;
    void removedFromAncestor(RemovalType, ContainerNode&) override;
    void didFinishInsertingNode() override;

    void clearTarget() override;
    void updateReferencedText(Element*);
    void detachTarget();
    void buildPendingResource() override;

    Ref<SVGTRefTargetEventListener> m_targetListener;
};

} // namespace WebCore
