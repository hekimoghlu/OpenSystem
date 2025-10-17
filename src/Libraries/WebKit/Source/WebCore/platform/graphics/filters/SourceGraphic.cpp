/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#include "SourceGraphic.h"

#include "Filter.h"
#include "SourceGraphicSoftwareApplier.h"
#include <wtf/text/TextStream.h>

#if USE(CORE_IMAGE)
#include "SourceGraphicCoreImageApplier.h"
#endif

#if USE(SKIA)
#include "SourceGraphicSkiaApplier.h"
#endif

namespace WebCore {

Ref<SourceGraphic> SourceGraphic::create(DestinationColorSpace colorSpace)
{
    return adoptRef(*new SourceGraphic(colorSpace));
}

SourceGraphic::SourceGraphic(DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::SourceGraphic, colorSpace)
{
}

OptionSet<FilterRenderingMode> SourceGraphic::supportedFilterRenderingModes() const
{
    OptionSet<FilterRenderingMode> modes = FilterRenderingMode::Software;
#if USE(CORE_IMAGE) || USE(SKIA)
    modes.add(FilterRenderingMode::Accelerated);
#endif
#if USE(GRAPHICS_CONTEXT_FILTERS)
    modes.add(FilterRenderingMode::GraphicsContext);
#endif
    return modes;
}

std::unique_ptr<FilterEffectApplier> SourceGraphic::createAcceleratedApplier() const
{
#if USE(CORE_IMAGE)
    return FilterEffectApplier::create<SourceGraphicCoreImageApplier>(*this);
#elif USE(SKIA)
    return FilterEffectApplier::create<SourceGraphicSkiaApplier>(*this);
#else
    return nullptr;
#endif
}

std::unique_ptr<FilterEffectApplier> SourceGraphic::createSoftwareApplier() const
{
#if USE(SKIA)
    return FilterEffectApplier::create<SourceGraphicSkiaApplier>(*this);
#else
    return FilterEffectApplier::create<SourceGraphicSoftwareApplier>(*this);
#endif
}

TextStream& SourceGraphic::externalRepresentation(TextStream& ts, FilterRepresentation) const
{
    ts << indent << "[SourceGraphic]\n";
    return ts;
}

} // namespace WebCore
