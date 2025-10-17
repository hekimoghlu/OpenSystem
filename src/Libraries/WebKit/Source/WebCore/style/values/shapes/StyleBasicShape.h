/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 10, 2021.
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

#include "CSSBasicShape.h"
#include "StyleCircleFunction.h"
#include "StyleEllipseFunction.h"
#include "StyleInsetFunction.h"
#include "StylePathComputation.h"
#include "StylePathFunction.h"
#include "StylePolygonFunction.h"
#include "StyleRectFunction.h"
#include "StyleShapeFunction.h"
#include "StyleWindRuleComputation.h"
#include "StyleXywhFunction.h"

namespace WebCore {
namespace Style {

// NOTE: This differs from CSS::BasicShape due to lack of RectFunction and XywhFunction, both of
// which convert to InsetFunction during style conversion.
using BasicShape = std::variant<
    CircleFunction,
    EllipseFunction,
    InsetFunction,
    PathFunction,
    PolygonFunction,
    ShapeFunction
>;

template<typename T> concept ShapeWithCenterCoordinate = std::same_as<T, CircleFunction> || std::same_as<T, EllipseFunction>;

// MARK: - Conversion

template<> struct ToCSS<BasicShape> { auto operator()(const BasicShape&, const RenderStyle&) -> CSS::BasicShape; };
template<> struct ToStyle<CSS::BasicShape> { auto operator()(const CSS::BasicShape&, const BuilderState&) -> BasicShape; };

// MARK: - Blending

template<> struct Blending<BasicShape> {
    auto canBlend(const BasicShape&, const BasicShape&) -> bool;
    auto blend(const BasicShape&, const BasicShape&, const BlendingContext&) -> BasicShape;
};

// MARK: - Path

template<> struct PathComputation<BasicShape> { WebCore::Path operator()(const BasicShape&, const FloatRect&); };

// MARK: - Winding

template<> struct WindRuleComputation<BasicShape> { WebCore::WindRule operator()(const BasicShape&); };

} // namespace Style
} // namespace WebCore
