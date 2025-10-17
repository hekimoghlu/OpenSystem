/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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

#include "CachedImageClient.h"
#include "CachedResourceHandle.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include "SVGURIReference.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGFEImageElement final : public SVGFilterPrimitiveStandardAttributes, public SVGURIReference, private CachedImageClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEImageElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEImageElement);
public:
    static Ref<SVGFEImageElement> create(const QualifiedName&, Document&);

    virtual ~SVGFEImageElement();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const { return SVGFilterPrimitiveStandardAttributes::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const { return SVGFilterPrimitiveStandardAttributes::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const { SVGFilterPrimitiveStandardAttributes::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const { SVGFilterPrimitiveStandardAttributes::decrementCheckedPtrCount(); }

    bool renderingTaintsOrigin() const;

    const SVGPreserveAspectRatioValue& preserveAspectRatio() const { return m_preserveAspectRatio->currentValue(); }
    SVGAnimatedPreserveAspectRatio& preserveAspectRatioAnimated() { return m_preserveAspectRatio; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEImageElement, SVGFilterPrimitiveStandardAttributes, SVGURIReference>;

private:
    SVGFEImageElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;
    void addSubresourceAttributeURLs(ListHashSet<URL>&) const override;

    void didFinishInsertingNode() override;

    std::tuple<RefPtr<ImageBuffer>, FloatRect> imageBufferForEffect(const GraphicsContext& destinationContext) const;

    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    void clearResourceReferences();
    void requestImageResource();

    void buildPendingResource() override;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) override;
    void removedFromAncestor(RemovalType, ContainerNode&) override;

    Ref<SVGAnimatedPreserveAspectRatio> m_preserveAspectRatio { SVGAnimatedPreserveAspectRatio::create(this) };
    CachedResourceHandle<CachedImage> m_cachedImage;
};

} // namespace WebCore
