/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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

#include "DisplayList.h"
#include "DisplayListRecorderImpl.h"
#include "GraphicsContext.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace DisplayList {

class DrawingContext {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(DrawingContext, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT DrawingContext(const FloatSize& logicalSize, const AffineTransform& initialCTM = { }, const DestinationColorSpace& = DestinationColorSpace::SRGB());
    WEBCORE_EXPORT ~DrawingContext();

    GraphicsContext& context() const { return const_cast<DrawingContext&>(*this).m_context; }
    RecorderImpl& recorder() { return m_context; };
    DisplayList& displayList() { return m_displayList; }
    const DisplayList& displayList() const { return m_displayList; }
    const DisplayList* replayedDisplayList() const { return m_replayedDisplayList.get(); }

    WEBCORE_EXPORT void setTracksDisplayListReplay(bool);
    WEBCORE_EXPORT void replayDisplayList(GraphicsContext&);

protected:
    RecorderImpl m_context;
    DisplayList m_displayList;
    std::unique_ptr<DisplayList> m_replayedDisplayList;
    bool m_tracksDisplayListReplay { false };
};

} // DisplayList
} // WebCore
