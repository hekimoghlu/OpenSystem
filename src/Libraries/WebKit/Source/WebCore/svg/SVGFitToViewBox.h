/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#include "FloatRect.h"
#include "QualifiedName.h"
#include "SVGAnimatedPropertyImpl.h"
#include "SVGNames.h"
#include "SVGPreserveAspectRatio.h"
#include "SVGPropertyOwnerRegistry.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AffineTransform;

class SVGFitToViewBox {
    WTF_MAKE_TZONE_ALLOCATED(SVGFitToViewBox);
    WTF_MAKE_NONCOPYABLE(SVGFitToViewBox);
public:
    static AffineTransform viewBoxToViewTransform(const FloatRect& viewBoxRect, const SVGPreserveAspectRatioValue&, float viewWidth, float viewHeight);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFitToViewBox>;

    const FloatRect& viewBox() const { return m_viewBox->currentValue(); }
    const SVGPreserveAspectRatioValue& preserveAspectRatio() const { return m_preserveAspectRatio->currentValue(); }

    SVGAnimatedRect& viewBoxAnimated() { return m_viewBox; }
    SVGAnimatedPreserveAspectRatio& preserveAspectRatioAnimated() { return m_preserveAspectRatio; }

    void setViewBox(const FloatRect&);
    void resetViewBox();

    void setPreserveAspectRatio(const SVGPreserveAspectRatioValue& preserveAspectRatio) { m_preserveAspectRatio->setBaseValInternal(preserveAspectRatio); }
    void resetPreserveAspectRatio() { m_preserveAspectRatio->setBaseValInternal({ }); }

    String viewBoxString() const { return SVGPropertyTraits<FloatRect>::toString(viewBox()); }
    String preserveAspectRatioString() const { return preserveAspectRatio().valueAsString(); }

    bool hasValidViewBox() const { return m_isViewBoxValid; }
    bool hasEmptyViewBox() const { return m_isViewBoxValid && viewBox().isEmpty(); }

protected:
    SVGFitToViewBox(SVGElement* contextElement, SVGPropertyAccess = SVGPropertyAccess::ReadWrite);

    static bool isKnownAttribute(const QualifiedName& attributeName) { return PropertyRegistry::isKnownAttribute(attributeName); }

    void reset();
    bool parseAttribute(const QualifiedName&, const AtomString&);
    std::optional<FloatRect> parseViewBox(StringView);
    std::optional<FloatRect> parseViewBox(StringParsingBuffer<LChar>&, bool validate = true);
    std::optional<FloatRect> parseViewBox(StringParsingBuffer<UChar>&, bool validate = true);

private:
    template<typename CharacterType> std::optional<FloatRect> parseViewBoxGeneric(StringParsingBuffer<CharacterType>&, bool validate = true);

    Ref<SVGAnimatedRect> m_viewBox;
    Ref<SVGAnimatedPreserveAspectRatio> m_preserveAspectRatio;
    bool m_isViewBoxValid { false };
};

} // namespace WebCore
