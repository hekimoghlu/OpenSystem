/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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
#include "config.h"
#include "FEMerge.h"

#include "FEMergeSoftwareApplier.h"
#include "ImageBuffer.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEMerge> FEMerge::create(unsigned numberOfEffectInputs, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEMerge(numberOfEffectInputs, colorSpace));
}

FEMerge::FEMerge(unsigned numberOfEffectInputs, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEMerge, colorSpace)
    , m_numberOfEffectInputs(numberOfEffectInputs)
{
}

bool FEMerge::operator==(const FEMerge& other) const
{
    return FilterEffect::operator==(other) && m_numberOfEffectInputs == other.m_numberOfEffectInputs;
}

std::unique_ptr<FilterEffectApplier> FEMerge::createSoftwareApplier() const
{
    return FilterEffectApplier::create<FEMergeSoftwareApplier>(*this);
}

TextStream& FEMerge::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feMerge";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " mergeNodes=\"" << m_numberOfEffectInputs << "\"";

    ts << "]\n";
    return ts;
}

} // namespace WebCore
