/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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

#include "FilterEffect.h"
#include "GraphicsTypes.h"

namespace WebCore {

class FEBlend : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEBlend> create(BlendMode, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEBlend&) const;

    BlendMode blendMode() const { return m_mode; }
    bool setBlendMode(BlendMode);

private:
    FEBlend(BlendMode, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEBlend>(*this, other); }

    unsigned numberOfEffectInputs() const override { return 2; }

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    BlendMode m_mode;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEBlend)
