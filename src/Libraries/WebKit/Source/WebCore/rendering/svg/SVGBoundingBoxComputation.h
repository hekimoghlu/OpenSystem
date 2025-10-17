/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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

#include "RenderLayerModelObject.h"
#include <wtf/Noncopyable.h>
#include <wtf/OptionSet.h>

namespace WebCore {

class FloatRect;

class SVGBoundingBoxComputation {
    WTF_MAKE_NONCOPYABLE(SVGBoundingBoxComputation);
public:
    explicit SVGBoundingBoxComputation(const RenderLayerModelObject&);
    ~SVGBoundingBoxComputation() = default;

    enum class DecorationOption : uint16_t {
        IncludeFillShape                    = 1 << 0, /* corresponds to 'bool fill'     */
        IncludeStrokeShape                  = 1 << 1, /* corresponds to 'bool stroke'   */
        IncludeMarkers                      = 1 << 2, /* corresponds to 'bool markers'  */
        IncludeClippers                     = 1 << 3, /* corresponds to 'bool clippers' */
        IncludeMaskers                      = 1 << 4, /* WebKit extension - internal    */
        IncludeOutline                      = 1 << 5, /* WebKit extension - internal    */
        IgnoreTransformations               = 1 << 6, /* WebKit extension - internal    */
        OverrideBoxWithFilterBox            = 1 << 7, /* WebKit extension - internal    */
        OverrideBoxWithFilterBoxForChildren = 1 << 8, /* WebKit extension - internal    */
        CalculateFastRepaintRect            = 1 << 9, /* WebKit extension - internal    */
        UseFilterBoxOnEmptyRect             = 1 << 10  /* WebKit extension - internal    */
    };

    using DecorationOptions = OptionSet<DecorationOption>;

    static constexpr DecorationOptions objectBoundingBoxDecoration = { DecorationOption::IncludeFillShape };
    static constexpr DecorationOptions strokeBoundingBoxDecoration = { DecorationOption::IncludeFillShape, DecorationOption::IncludeStrokeShape };
    static constexpr DecorationOptions filterBoundingBoxDecoration = { DecorationOption::OverrideBoxWithFilterBox, DecorationOption::OverrideBoxWithFilterBoxForChildren };
    static constexpr DecorationOptions repaintBoundingBoxDecoration = { DecorationOption::IncludeFillShape, DecorationOption::IncludeStrokeShape, DecorationOption::IncludeMarkers, DecorationOption::IncludeClippers, DecorationOption::IncludeMaskers, DecorationOption::OverrideBoxWithFilterBox, DecorationOption::CalculateFastRepaintRect };

    FloatRect computeDecoratedBoundingBox(const DecorationOptions&, bool* boundingBoxValid = nullptr) const;

    static FloatRect computeDecoratedBoundingBox(const RenderLayerModelObject& renderer, const DecorationOptions& options)
    {
        SVGBoundingBoxComputation boundingBoxComputation(renderer);
        return boundingBoxComputation.computeDecoratedBoundingBox(options);
    }

    static FloatRect computeRepaintBoundingBox(const RenderLayerModelObject& renderer)
    {
        return computeDecoratedBoundingBox(renderer, repaintBoundingBoxDecoration);
    }

    static LayoutRect computeVisualOverflowRect(const RenderLayerModelObject&);

private:
    FloatRect handleShapeOrTextOrInline(const DecorationOptions&, bool* boundingBoxValid = nullptr) const;
    FloatRect handleRootOrContainer(const DecorationOptions&, bool* boundingBoxValid = nullptr) const;
    FloatRect handleForeignObjectOrImage(const DecorationOptions&, bool* boundingBoxValid = nullptr) const;

    void adjustBoxForClippingAndEffects(const DecorationOptions&, FloatRect& box, const DecorationOptions& optionsToCheckForFilters = filterBoundingBoxDecoration) const;

    SingleThreadWeakRef<const RenderLayerModelObject> m_renderer;
};

} // namespace WebCore
