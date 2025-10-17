/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#include "DrawGlyphsRecorder.h"

#include "FloatPoint.h"
#include "Font.h"
#include "GlyphBuffer.h"

namespace WebCore {

DrawGlyphsRecorder::DrawGlyphsRecorder(GraphicsContext& owner, float, DeriveFontFromContext deriveFontFromContext)
    : m_owner(owner)
    , m_deriveFontFromContext(deriveFontFromContext)
{
}

void DrawGlyphsRecorder::drawGlyphs(const Font& font, std::span<const GlyphBufferGlyph> glyphs, std::span<const GlyphBufferAdvance> advances, const FloatPoint& startPoint, FontSmoothingMode smoothingMode)
{
    m_owner.drawGlyphsAndCacheResources(font, glyphs, advances, startPoint, smoothingMode);
}

} // namespace WebCore
