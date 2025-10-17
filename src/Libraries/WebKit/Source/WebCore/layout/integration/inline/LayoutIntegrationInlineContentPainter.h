/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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

#include "LayoutIntegrationBoxTreeUpdater.h"
#include "LayoutPoint.h"
#include "LayoutRect.h"
#include <wtf/WeakListHashSet.h>

namespace WebCore {

class RenderBlock;
class RenderBlockFlow;
class RenderBox;
class RenderInline;

struct PaintInfo;

namespace InlineDisplay {
struct Box;
};

namespace Layout {
class ElementBox;
};

namespace LayoutIntegration {

struct InlineContent;

class InlineContentPainter {
public:
    InlineContentPainter(PaintInfo&, const LayoutPoint& paintOffset, const RenderInline* inlineBoxWithLayer, const InlineContent&, const RenderBlockFlow& root);

    void paint();

private:
    void paintDisplayBox(const InlineDisplay::Box&);
    void paintEllipsis(size_t lineIndex);
    LayoutPoint flippedContentOffsetIfNeeded(const RenderBox&) const;
    const RenderBlock& root() const;

    PaintInfo& m_paintInfo;
    const LayoutPoint m_paintOffset;
    LayoutRect m_damageRect;
    const RenderInline* m_inlineBoxWithLayer { nullptr };
    const InlineContent& m_inlineContent;
    const RenderBlockFlow& m_root;
    SingleThreadWeakListHashSet<RenderInline> m_outlineObjects;
};

class LayerPaintScope {
public:
    LayerPaintScope(const RenderInline* inlineBoxWithLayer);
    bool includes(const InlineDisplay::Box&);

private:
    const Layout::ElementBox* const m_inlineBoxWithLayer;
    const Layout::ElementBox* m_currentExcludedInlineBox { nullptr };
};

}
}

