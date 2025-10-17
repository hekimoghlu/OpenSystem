/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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

#include "TextureMapperFPSCounter.h"

#include "TextureMapper.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextureMapperFPSCounter);

TextureMapperFPSCounter::TextureMapperFPSCounter()
    : m_isShowingFPS(false)
    , m_fpsInterval(0_s)
    , m_lastFPS(0)
    , m_frameCount(0)
{
    auto showFPSEnvironment = String::fromLatin1(getenv("WEBKIT_SHOW_FPS"));
    bool ok = false;
    m_fpsInterval = Seconds(showFPSEnvironment.toDouble(&ok));
    if (ok && m_fpsInterval) {
        m_isShowingFPS = true;
        m_fpsTimestamp = MonotonicTime::now();
    }
}

void TextureMapperFPSCounter::updateFPSAndDisplay(TextureMapper& textureMapper, const FloatPoint& location, const TransformationMatrix& matrix)
{
    if (!m_isShowingFPS)
        return;

    m_frameCount++;
    Seconds delta = MonotonicTime::now() - m_fpsTimestamp;
    if (delta >= m_fpsInterval) {
        m_lastFPS = int(m_frameCount / delta.seconds());
        m_frameCount = 0;
        m_fpsTimestamp += delta;
    }

    textureMapper.drawNumber(m_lastFPS, Color::black, location, matrix);
}

} // namespace WebCore
