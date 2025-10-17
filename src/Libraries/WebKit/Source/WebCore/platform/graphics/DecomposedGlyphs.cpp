/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 14, 2022.
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
#include "DecomposedGlyphs.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DecomposedGlyphs);

Ref<DecomposedGlyphs> DecomposedGlyphs::create(std::span<const GlyphBufferGlyph> glyphs, std::span<const GlyphBufferAdvance> advances, const FloatPoint& localAnchor, FontSmoothingMode mode, RenderingResourceIdentifier renderingResourceIdentifier)
{
    return adoptRef(*new DecomposedGlyphs({ Vector(glyphs), Vector(advances), localAnchor, mode }, renderingResourceIdentifier));
}

Ref<DecomposedGlyphs> DecomposedGlyphs::create(PositionedGlyphs&& positionedGlyphs, RenderingResourceIdentifier renderingResourceIdentifier)
{
    return adoptRef(*new DecomposedGlyphs(WTFMove(positionedGlyphs), renderingResourceIdentifier));
}

DecomposedGlyphs::DecomposedGlyphs(PositionedGlyphs&& positionedGlyphs, RenderingResourceIdentifier renderingResourceIdentifier)
    : RenderingResource(renderingResourceIdentifier)
    , m_positionedGlyphs(WTFMove(positionedGlyphs))
{
    ASSERT(m_positionedGlyphs.glyphs.size() == m_positionedGlyphs.advances.size());
}

} // namespace WebCore
