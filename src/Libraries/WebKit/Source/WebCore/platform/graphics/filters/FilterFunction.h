/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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

#include "FilterEffectGeometry.h"
#include "FilterImage.h"
#include "FilterImageVector.h"
#include "FilterRenderingMode.h"
#include "FilterStyle.h"
#include "FloatRect.h"
#include "LengthBox.h"
#include "RenderingResource.h"
#include <wtf/text/AtomString.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class Filter;
class FilterResults;
class GraphicsContext;

enum class FilterRepresentation : uint8_t {
    TestOutput,
    Debugging
};

class FilterFunction : public RenderingResource {
public:
    enum class Type : uint8_t {
        CSSFilter,
        SVGFilter,

        // These are filter effects
        FEBlend,
        FEColorMatrix,
        FEComponentTransfer,
        FEComposite,
        FEConvolveMatrix,
        FEDiffuseLighting,
        FEDisplacementMap,
        FEDropShadow,
        FEFlood,
        FEGaussianBlur,
        FEImage,
        FEMerge,
        FEMorphology,
        FEOffset,
        FESpecularLighting,
        FETile,
        FETurbulence,
        SourceAlpha,
        SourceGraphic
    };

    FilterFunction(Type, std::optional<RenderingResourceIdentifier> = std::nullopt);
    virtual ~FilterFunction() = default;

    Type filterType() const { return m_filterType; }

    bool isCSSFilter() const { return m_filterType == Type::CSSFilter; }
    bool isSVGFilter() const { return m_filterType == Type::SVGFilter; }
    bool isFilter() const override { return m_filterType == Type::CSSFilter || m_filterType == Type::SVGFilter; }
    bool isFilterEffect() const { return m_filterType >= Type::FEBlend && m_filterType <= Type::SourceGraphic; }

    static AtomString filterName(Type);
    static AtomString sourceAlphaName() { return filterName(Type::SourceAlpha); }
    static AtomString sourceGraphicName() { return filterName(Type::SourceGraphic); }
    AtomString filterName() const { return filterName(m_filterType); }

    virtual OptionSet<FilterRenderingMode> supportedFilterRenderingModes() const { return FilterRenderingMode::Software; }
    virtual RefPtr<FilterImage> apply(const Filter&, FilterImage&, FilterResults&) { return nullptr; }
    virtual FilterStyleVector createFilterStyles(GraphicsContext&, const Filter&, const FilterStyle&) const { return { }; }

    virtual WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation = FilterRepresentation::TestOutput) const = 0;

private:
    Type m_filterType;
};

WEBCORE_EXPORT TextStream& operator<<(TextStream&, const FilterFunction&);

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(ClassName) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ClassName) \
    static bool isType(const WebCore::FilterFunction& filter) { return filter.filterType() == WebCore::FilterFunction::Type::ClassName; } \
SPECIALIZE_TYPE_TRAITS_END()
