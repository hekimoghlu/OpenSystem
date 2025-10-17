/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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

#include "CanvasRenderingContext2DBase.h"
#include "HTMLCanvasElement.h"
#include <memory>

namespace WebCore {

class TextMetrics;

class CanvasRenderingContext2D final : public CanvasRenderingContext2DBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CanvasRenderingContext2D);
public:
    static std::unique_ptr<CanvasRenderingContext2D> create(CanvasBase&, CanvasRenderingContext2DSettings&&, bool usesCSSCompatibilityParseMode);

    virtual ~CanvasRenderingContext2D();

    HTMLCanvasElement& canvas() const { return downcast<HTMLCanvasElement>(canvasBase()); }

    void drawFocusIfNeeded(Element&);
    void drawFocusIfNeeded(Path2D&, Element&);

    void setFont(const String&);

    CanvasDirection direction() const;

    void fillText(const String& text, double x, double y, std::optional<double> maxWidth = std::nullopt);
    void strokeText(const String& text, double x, double y, std::optional<double> maxWidth = std::nullopt);
    Ref<TextMetrics> measureText(const String& text);

private:
    CanvasRenderingContext2D(CanvasBase&, CanvasRenderingContext2DSettings&&, bool usesCSSCompatibilityParseMode);

    const FontProxy* fontProxy() final;

    std::optional<FilterOperations> setFilterStringWithoutUpdatingStyle(const String&) override;
    RefPtr<Filter> createFilter(const FloatRect& bounds) const override;
    IntOutsets calculateFilterOutsets(const FloatRect& bounds) const override;

    void setFontWithoutUpdatingStyle(const String&);

    void drawTextInternal(const String& text, double x, double y, bool fill, std::optional<double> maxWidth = std::nullopt);

    void drawFocusIfNeededInternal(const Path&, Element&);

    TextDirection toTextDirection(CanvasRenderingContext2DBase::Direction, const RenderStyle** computedStyle = nullptr) const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CANVASRENDERINGCONTEXT(WebCore::CanvasRenderingContext2D, is2d())
