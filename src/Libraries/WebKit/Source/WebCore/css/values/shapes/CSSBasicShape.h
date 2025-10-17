/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

#include "CSSCircleFunction.h"
#include "CSSEllipseFunction.h"
#include "CSSInsetFunction.h"
#include "CSSPathFunction.h"
#include "CSSPolygonFunction.h"
#include "CSSRectFunction.h"
#include "CSSShapeFunction.h"
#include "CSSXywhFunction.h"

namespace WebCore {
namespace CSS {

// <basic-shape> = <circle()> | <ellipse() | <inset()> | <path()> | <polygon()> | <rect()> | <shape()> | <xywh()>
// https://drafts.csswg.org/css-shapes/#typedef-basic-shape
using BasicShape = std::variant<
    CircleFunction,
    EllipseFunction,
    InsetFunction,
    PathFunction,
    PolygonFunction,
    RectFunction,
    ShapeFunction,
    XywhFunction
>;

} // namespace CSS
} // namespace WebCore
