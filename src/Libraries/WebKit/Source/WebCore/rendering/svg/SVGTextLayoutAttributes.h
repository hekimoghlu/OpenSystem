/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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

#include "SVGTextMetrics.h"
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class RenderSVGInlineText;

struct SVGCharacterData {
    SVGCharacterData();

    float x;
    float y;
    float dx;
    float dy;
    float rotate;
};

typedef UncheckedKeyHashMap<unsigned, SVGCharacterData> SVGCharacterDataMap;

class SVGTextLayoutAttributes {
    WTF_MAKE_NONCOPYABLE(SVGTextLayoutAttributes);
public:
    explicit SVGTextLayoutAttributes(RenderSVGInlineText&);

    void clear();
    static constexpr float emptyValue() { return std::numeric_limits<float>::quiet_NaN(); }
    static bool isEmptyValue(float value) { return std::isnan(value); }

    RenderSVGInlineText& context();
    const RenderSVGInlineText& context() const;
    
    SVGCharacterDataMap& characterDataMap() { return m_characterDataMap; }
    const SVGCharacterDataMap& characterDataMap() const { return m_characterDataMap; }

    Vector<SVGTextMetrics>& textMetricsValues() { return m_textMetricsValues; }
    const Vector<SVGTextMetrics>& textMetricsValues() const { return m_textMetricsValues; }

private:
    SingleThreadWeakRef<RenderSVGInlineText> m_context;
    SVGCharacterDataMap m_characterDataMap;
    Vector<SVGTextMetrics> m_textMetricsValues;
};

inline SVGCharacterData::SVGCharacterData()
    : x(SVGTextLayoutAttributes::emptyValue())
    , y(SVGTextLayoutAttributes::emptyValue())
    , dx(SVGTextLayoutAttributes::emptyValue())
    , dy(SVGTextLayoutAttributes::emptyValue())
    , rotate(SVGTextLayoutAttributes::emptyValue())
{
}

} // namespace WebCore
