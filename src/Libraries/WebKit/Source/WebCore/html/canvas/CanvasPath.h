/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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

#include "ExceptionOr.h"
#include "Path.h"
#include <variant>
#include <wtf/Forward.h>

namespace WebCore {

struct DOMPointInit;

class CanvasPath {
public:
    using RadiusVariant = std::variant<double, DOMPointInit>;
    virtual ~CanvasPath() = default;

    void closePath();
    void moveTo(float x, float y);
    void lineTo(float x, float y);
    void quadraticCurveTo(float cpx, float cpy, float x, float y);
    void bezierCurveTo(float cp1x, float cp1y, float cp2x, float cp2y, float x, float y);
    ExceptionOr<void> arcTo(float x0, float y0, float x1, float y1, float radius);
    ExceptionOr<void> arc(float x, float y, float r, float sa, float ea, bool anticlockwise);
    ExceptionOr<void> ellipse(float x, float y, float radiusX, float radiusY, float rotation, float startAngle, float endAngled, bool anticlockwise);
    void rect(float x, float y, float width, float height);
    ExceptionOr<void> roundRect(float x, float y, float width, float height, const RadiusVariant& radii);
    ExceptionOr<void> roundRect(float x, float y, float width, float height, std::span<const RadiusVariant> radii);

    float currentX() const;
    float currentY() const;

protected:
    CanvasPath() = default;
    CanvasPath(const Path& path)
        : m_path(path)
    { }

    virtual bool hasInvertibleTransform() const { return true; }

    void lineTo(FloatPoint);

    Path m_path;
};

}
