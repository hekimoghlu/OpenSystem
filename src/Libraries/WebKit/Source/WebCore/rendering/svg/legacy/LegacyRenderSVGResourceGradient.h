/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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

#include "ImageBuffer.h"
#include "LegacyRenderSVGResourceContainer.h"
#include "SVGGradientElement.h"
#include <memory>
#include <wtf/HashMap.h>

namespace WebCore {

class GraphicsContext;

struct GradientData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    struct Inputs {
        friend bool operator==(const Inputs&, const Inputs&) = default;

        std::optional<FloatRect> objectBoundingBox;
        float textPaintingScale = 1;
    };

    bool invalidate(const Inputs& inputs)
    {
        if (this->inputs != inputs) {
            gradient = nullptr;
            userspaceTransform = AffineTransform();
            this->inputs = inputs;
        }
        return !gradient;
    }

    RefPtr<Gradient> gradient;
    AffineTransform userspaceTransform;
    Inputs inputs;
};

class GradientApplier {
public:
    GradientApplier() = default;
    virtual ~GradientApplier() = default;

    virtual bool applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, const GradientData&, OptionSet<RenderSVGResourceMode>) = 0;
    virtual void postApplyResource(RenderElement&, GraphicsContext*&, const GradientData&, SVGUnitTypes::SVGUnitType gradientUnits, const AffineTransform& gradientTransform, OptionSet<RenderSVGResourceMode>, const Path*, const RenderElement*) = 0;
};

class LegacyRenderSVGResourceGradient : public LegacyRenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourceGradient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourceGradient);
public:
    virtual ~LegacyRenderSVGResourceGradient();

    SVGGradientElement& gradientElement() const { return static_cast<SVGGradientElement&>(LegacyRenderSVGResourceContainer::element()); }

    void removeAllClientsFromCache() final;
    void removeAllClientsFromCacheAndMarkForInvalidationIfNeeded(bool markForInvalidation, SingleThreadWeakHashSet<RenderObject>* visitedRenderers) override;
    void removeClientFromCache(RenderElement&) final;

    OptionSet<ApplyResult> applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>) final;
    void postApplyResource(RenderElement&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>, const Path*, const RenderElement*) final;
    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation) final { return FloatRect(); }

protected:
    LegacyRenderSVGResourceGradient(Type, SVGGradientElement&, RenderStyle&&);

    static GradientColorStops stopsByApplyingColorFilter(const GradientColorStops&, const RenderStyle&);
    static GradientSpreadMethod platformSpreadMethodFromSVGType(SVGSpreadMethodType);

private:
    void element() const = delete;

    GradientData::Inputs computeInputs(RenderElement&, OptionSet<RenderSVGResourceMode>);
    GradientData* gradientDataForRenderer(RenderElement&, const RenderStyle&, OptionSet<RenderSVGResourceMode>);

    virtual SVGUnitTypes::SVGUnitType gradientUnits() const = 0;
    virtual AffineTransform gradientTransform() const = 0;
    virtual bool collectGradientAttributes() = 0;
    virtual Ref<Gradient> buildGradient(const RenderStyle&) const = 0;

    UncheckedKeyHashMap<RenderObject*, std::unique_ptr<GradientData>> m_gradientMap;

    std::unique_ptr<GradientApplier> m_gradientApplier;
    bool m_shouldCollectGradientAttributes { true };
};

} // namespace WebCore
