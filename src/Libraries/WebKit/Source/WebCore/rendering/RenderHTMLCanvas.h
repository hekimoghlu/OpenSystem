/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#include "RenderReplaced.h"

namespace WebCore {

class HTMLCanvasElement;

class RenderHTMLCanvas final : public RenderReplaced {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderHTMLCanvas);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderHTMLCanvas);
public:
    RenderHTMLCanvas(HTMLCanvasElement&, RenderStyle&&);
    virtual ~RenderHTMLCanvas();

    HTMLCanvasElement& canvasElement() const;

    void canvasSizeChanged();

private:
    void element() const = delete;
    bool requiresLayer() const override;
    ASCIILiteral renderName() const override { return "RenderHTMLCanvas"_s; }
    void paintReplaced(PaintInfo&, const LayoutPoint&) override;
    void intrinsicSizeChanged() override { canvasSizeChanged(); }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderHTMLCanvas, isRenderHTMLCanvas())
