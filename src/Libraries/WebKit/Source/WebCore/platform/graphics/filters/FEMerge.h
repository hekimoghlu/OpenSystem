/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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

namespace WebCore {

class FEMerge : public FilterEffect {
public:
    WEBCORE_EXPORT static Ref<FEMerge> create(unsigned numberOfEffectInputs, DestinationColorSpace = DestinationColorSpace::SRGB());

    bool operator==(const FEMerge&) const;

    unsigned numberOfEffectInputs() const override { return m_numberOfEffectInputs; }

private:
    FEMerge(unsigned numberOfEffectInputs, DestinationColorSpace);

    bool operator==(const FilterEffect& other) const override { return areEqual<FEMerge>(*this, other); }

    std::unique_ptr<FilterEffectApplier> createSoftwareApplier() const override;

    WTF::TextStream& externalRepresentation(WTF::TextStream&, FilterRepresentation) const override;

    unsigned m_numberOfEffectInputs { 0 };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_FILTER_FUNCTION(FEMerge)
