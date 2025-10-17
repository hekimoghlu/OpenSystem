/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

#include "RenderElement.h"

namespace WebCore {

class RenderElement;
class RenderLayerModelObject;

class LayoutRepainter {
public:
    enum class CheckForRepaint : uint8_t { No, Yes };
    enum class ShouldAlwaysIssueFullRepaint : uint8_t { No, Yes };
    LayoutRepainter(RenderElement&, std::optional<CheckForRepaint> checkForRepaintOverride = { }, std::optional<ShouldAlwaysIssueFullRepaint> = { }, RepaintOutlineBounds = RepaintOutlineBounds::Yes);

    // Return true if it repainted.
    bool repaintAfterLayout();

private:
    CheckedRef<RenderElement> m_renderer;
    const RenderLayerModelObject* m_repaintContainer { nullptr };
    // We store these values as LayoutRects, but the final invalidations will be pixel snapped
    RenderObject::RepaintRects m_oldRects;
    bool m_checkForRepaint { true };
    bool m_forceFullRepaint { false };
    RepaintOutlineBounds m_repaintOutlineBounds;
};

} // namespace WebCore
