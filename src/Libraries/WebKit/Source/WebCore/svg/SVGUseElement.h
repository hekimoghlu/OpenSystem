/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

#include "CachedResourceHandle.h"
#include "CachedSVGDocumentClient.h"
#include "SVGGraphicsElement.h"
#include "SVGURIReference.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CachedSVGDocument;

class SVGUseElement final : public SVGGraphicsElement, public SVGURIReference, private CachedSVGDocumentClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGUseElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGUseElement);
public:
    static Ref<SVGUseElement> create(const QualifiedName&, Document&);
    virtual ~SVGUseElement();

    void invalidateShadowTree();
    void updateUserAgentShadowTree() final;

    RefPtr<SVGElement> clipChild() const;
    RenderElement* rendererClipChild() const;

    const SVGLengthValue& x() const { return m_x->currentValue(); }
    const SVGLengthValue& y() const { return m_y->currentValue(); }
    const SVGLengthValue& width() const { return m_width->currentValue(); }
    const SVGLengthValue& height() const { return m_height->currentValue(); }

    SVGAnimatedLength& xAnimated() { return m_x; }
    SVGAnimatedLength& yAnimated() { return m_y; }
    SVGAnimatedLength& widthAnimated() { return m_width; }
    SVGAnimatedLength& heightAnimated() { return m_height; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGUseElement, SVGGraphicsElement, SVGURIReference>;

private:
    SVGUseElement(const QualifiedName&, Document&);

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) override;
    void didFinishInsertingNode() final;
    void removedFromAncestor(RemovalType, ContainerNode&) override;
    void buildPendingResource() override;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    Path toClipPath() override;
    bool selfHasRelativeLengths() const override;
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;

    Document* externalDocument() const;
    void updateExternalDocument();

    RefPtr<SVGElement> findTarget(AtomString* targetID = nullptr) const;

    void cloneTarget(ContainerNode&, SVGElement& target) const;
    RefPtr<SVGElement> targetClone() const;

    void expandUseElementsInShadowTree() const;
    void expandSymbolElementsInShadowTree() const;
    void transferEventListenersToShadowTree() const;
    void transferSizeAttributesToTargetClone(SVGElement&) const;

    void clearShadowTree();
    void invalidateDependentShadowTrees();

    bool haveLoadedRequiredResources() override { return SVGURIReference::haveLoadedRequiredResources(); }
    void setHaveFiredLoadEvent(bool haveFiredLoadEvent) override { m_haveFiredLoadEvent = haveFiredLoadEvent; }
    bool haveFiredLoadEvent() const override { return m_haveFiredLoadEvent; }
    void setErrorOccurred(bool errorOccurred) override { m_errorOccurred = errorOccurred; }
    bool errorOccurred() const override { return m_errorOccurred; }
    Timer* loadEventTimer() override { return &m_loadEventTimer; }

    bool isValid() const override { return SVGTests::isValid(); }

    Ref<SVGAnimatedLength> m_x { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_y { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_width { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_height { SVGAnimatedLength::create(this, SVGLengthMode::Height) };

    bool m_haveFiredLoadEvent { false };
    bool m_errorOccurred { false };
    bool m_shadowTreeNeedsUpdate { true };
    CachedResourceHandle<CachedSVGDocument> m_externalDocument;
    Timer m_loadEventTimer;
};

}
