/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

#if ENABLE(OFFSCREEN_CANVAS)

#include "CanvasRenderingContext2DBase.h"

#include "OffscreenCanvas.h"

namespace WebCore {

class OffscreenCanvasRenderingContext2D final : public CanvasRenderingContext2DBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(OffscreenCanvasRenderingContext2D);
public:
    static bool enabledForContext(ScriptExecutionContext&);
    static std::unique_ptr<OffscreenCanvasRenderingContext2D> create(CanvasBase&, CanvasRenderingContext2DSettings&&);

    virtual ~OffscreenCanvasRenderingContext2D();

    OffscreenCanvas& canvas() const { return downcast<OffscreenCanvas>(canvasBase()); }

    void setFont(const String&);
    CanvasDirection direction() const;
    void fillText(const String& text, double x, double y, std::optional<double> maxWidth = std::nullopt);
    void strokeText(const String& text, double x, double y, std::optional<double> maxWidth = std::nullopt);
    Ref<TextMetrics> measureText(const String& text);

private:
    OffscreenCanvasRenderingContext2D(CanvasBase&, CanvasRenderingContext2DSettings&&);
    RefPtr<ImageBuffer> transferToImageBuffer() final;
    const FontProxy* fontProxy() final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CANVASRENDERINGCONTEXT(WebCore::OffscreenCanvasRenderingContext2D, isOffscreen2d())

#endif
