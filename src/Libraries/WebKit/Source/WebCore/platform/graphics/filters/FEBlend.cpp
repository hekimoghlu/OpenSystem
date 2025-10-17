/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#include "FEBlend.h"

#include "FEBlendNeonApplier.h"
#include "FEBlendSoftwareApplier.h"
#include "ImageBuffer.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<FEBlend> FEBlend::create(BlendMode mode, DestinationColorSpace colorSpace)
{
    return adoptRef(*new FEBlend(mode, colorSpace));
}

FEBlend::FEBlend(BlendMode mode, DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::FEBlend, colorSpace)
    , m_mode(mode)
{
}

bool FEBlend::operator==(const FEBlend& other) const
{
    return FilterEffect::operator==(other) && m_mode == other.m_mode;
}

bool FEBlend::setBlendMode(BlendMode mode)
{
    if (m_mode == mode)
        return false;
    m_mode = mode;
    return true;
}

std::unique_ptr<FilterEffectApplier> FEBlend::createSoftwareApplier() const
{
#if HAVE(ARM_NEON_INTRINSICS)
    return FilterEffectApplier::create<FEBlendNeonApplier>(*this);
#else
    return FilterEffectApplier::create<FEBlendSoftwareApplier>(*this);
#endif
}

TextStream& FEBlend::externalRepresentation(TextStream& ts, FilterRepresentation representation) const
{
    ts << indent << "[feBlend";
    FilterEffect::externalRepresentation(ts, representation);

    ts << " mode=\"" << (m_mode == BlendMode::Normal ? "normal"_s : compositeOperatorName(CompositeOperator::SourceOver, m_mode));

    ts << "\"]\n";
    return ts;
}

} // namespace WebCore
