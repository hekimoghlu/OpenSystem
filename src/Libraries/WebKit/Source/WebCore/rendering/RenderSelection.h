/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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

#include "RenderHighlight.h"
#include "RenderSelectionGeometry.h"
#if ENABLE(SERVICE_CONTROLS)
#include "SelectionGeometryGatherer.h"
#endif

namespace WebCore {

class RenderSelection : public RenderHighlight {
public:
    RenderSelection(RenderView&);
    
    enum class RepaintMode { NewXOROld, NewMinusOld, Nothing };
    void set(const RenderRange&, RepaintMode = RepaintMode::NewXOROld);
    void clear();
    void repaint() const;
    
    IntRect bounds() const { return collectBounds(ClipToVisibleContent::No); }
    IntRect boundsClippedToVisibleContent() const { return collectBounds(ClipToVisibleContent::Yes); }
    
private:
    const RenderView& m_renderView;
#if ENABLE(SERVICE_CONTROLS)
    SelectionGeometryGatherer m_selectionGeometryGatherer;
#endif
    bool m_selectionWasCaret { false };
    enum class ClipToVisibleContent : bool { No, Yes };
    IntRect collectBounds(ClipToVisibleContent) const;
    void apply(const RenderRange&, RepaintMode);
};

} // namespace WebCore
