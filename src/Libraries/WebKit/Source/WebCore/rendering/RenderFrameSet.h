/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

#include "RenderBox.h"

namespace WebCore {

class HTMLFrameSetElement;
class MouseEvent;
class RenderFrame;

enum FrameEdge { LeftFrameEdge, RightFrameEdge, TopFrameEdge, BottomFrameEdge };

struct FrameEdgeInfo {
    explicit FrameEdgeInfo(bool preventResize = false, bool allowBorder = true)
        : m_preventResize(4, preventResize)
        , m_allowBorder(4, allowBorder)
    {
    }

    bool preventResize(FrameEdge edge) const { return m_preventResize[edge]; }
    bool allowBorder(FrameEdge edge) const { return m_allowBorder[edge]; }

    void setPreventResize(FrameEdge edge, bool preventResize) { m_preventResize[edge] = preventResize; }
    void setAllowBorder(FrameEdge edge, bool allowBorder) { m_allowBorder[edge] = allowBorder; }

private:
    Vector<bool> m_preventResize;
    Vector<bool> m_allowBorder;
};

class RenderFrameSet final : public RenderBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderFrameSet);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderFrameSet);
public:
    RenderFrameSet(HTMLFrameSetElement&, RenderStyle&&);
    virtual ~RenderFrameSet();

    HTMLFrameSetElement& frameSetElement() const;

    FrameEdgeInfo edgeInfo() const;

    bool userResize(MouseEvent&);

    bool canResizeRow(const IntPoint&) const;
    bool canResizeColumn(const IntPoint&) const;

    void notifyFrameEdgeInfoChanged();

private:
    void element() const = delete;
    void computeIntrinsicLogicalWidths(LayoutUnit&, LayoutUnit&) const override { }

    static const int noSplit = -1;

    class GridAxis {
        WTF_MAKE_NONCOPYABLE(GridAxis);
    public:
        GridAxis();
        void resize(int);

        Vector<int> m_sizes;
        Vector<int> m_deltas;
        Vector<bool> m_preventResize;
        Vector<bool> m_allowBorder;
        int m_splitBeingResized;
        int m_splitResizeOffset;
    };

    ASCIILiteral renderName() const override { return "RenderFrameSet"_s; }

    void layout() override;
    void paint(PaintInfo&, const LayoutPoint&) override;
    bool canHaveChildren() const override { return true; }
    bool isChildAllowed(const RenderObject&, const RenderStyle&) const override;
    CursorDirective getCursor(const LayoutPoint&, Cursor&) const override;

    void setIsResizing(bool);

    void layOutAxis(GridAxis&, std::span<const Length>, int availableSpace);
    void computeEdgeInfo();
    void fillFromEdgeInfo(const FrameEdgeInfo& edgeInfo, int r, int c);
    void positionFrames();

    int splitPosition(const GridAxis&, int split) const;
    int hitTestSplit(const GridAxis&, int position) const;

    void startResizing(GridAxis&, int position);
    void continueResizing(GridAxis&, int position);

    void paintRowBorder(const PaintInfo&, const IntRect&);
    void paintColumnBorder(const PaintInfo&, const IntRect&);

    GridAxis m_rows;
    GridAxis m_cols;

    bool m_isResizing;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderFrameSet, isRenderFrameSet())
