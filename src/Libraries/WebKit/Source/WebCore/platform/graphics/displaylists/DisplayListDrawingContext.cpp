/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#include "DisplayListDrawingContext.h"

#include "AffineTransform.h"
#include "DisplayListRecorderImpl.h"
#include "DisplayListReplayer.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace DisplayList {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DrawingContext);

DrawingContext::DrawingContext(const FloatSize& logicalSize, const AffineTransform& initialCTM, const DestinationColorSpace& colorSpace)
    : m_context(m_displayList, GraphicsContextState(), FloatRect({ }, logicalSize), initialCTM, colorSpace)
{
}

DrawingContext::~DrawingContext() = default;

void DrawingContext::setTracksDisplayListReplay(bool tracksDisplayListReplay)
{
    m_tracksDisplayListReplay = tracksDisplayListReplay;
    m_replayedDisplayList.reset();
}

void DrawingContext::replayDisplayList(GraphicsContext& destContext)
{
    if (m_displayList.isEmpty())
        return;

    Replayer replayer(destContext, m_displayList);
    if (m_tracksDisplayListReplay)
        m_replayedDisplayList = replayer.replay({ }, m_tracksDisplayListReplay).trackedDisplayList;
    else
        replayer.replay();
    m_displayList.clear();
}

} // DisplayList
} // WebCore
