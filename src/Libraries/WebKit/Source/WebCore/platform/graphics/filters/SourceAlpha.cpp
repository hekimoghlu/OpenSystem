/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "SourceAlpha.h"

#include "ImageBuffer.h"
#include "SourceAlphaSoftwareApplier.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<SourceAlpha> SourceAlpha::create(const DestinationColorSpace& colorSpace)
{
    return adoptRef(*new SourceAlpha(colorSpace));
}

SourceAlpha::SourceAlpha(DestinationColorSpace colorSpace)
    : FilterEffect(FilterEffect::Type::SourceAlpha, colorSpace)
{
}

std::unique_ptr<FilterEffectApplier> SourceAlpha::createSoftwareApplier() const
{
    return FilterEffectApplier::create<SourceAlphaSoftwareApplier>(*this);
}

TextStream& SourceAlpha::externalRepresentation(TextStream& ts, FilterRepresentation) const
{
    ts << indent << "[SourceAlpha]\n";
    return ts;
}

} // namespace WebCore
