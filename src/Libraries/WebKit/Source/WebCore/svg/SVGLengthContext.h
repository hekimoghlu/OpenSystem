/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#include "FloatRect.h"
#include "SVGLengthValue.h"
#include "SVGUnitTypes.h"

namespace WebCore {

class SVGElement;
class WeakPtrImplWithEventTargetData;

struct Length;

class SVGLengthContext {
public:
    explicit SVGLengthContext(const SVGElement*);
    ~SVGLengthContext();

    template<typename T>
    static FloatRect resolveRectangle(const T* context, SVGUnitTypes::SVGUnitType type, const FloatRect& viewport)
    {
        return resolveRectangle(context, type, viewport, context->x(), context->y(), context->width(), context->height());
    }

    static FloatRect resolveRectangle(const SVGElement*, SVGUnitTypes::SVGUnitType, const FloatRect& viewport, const SVGLengthValue& x, const SVGLengthValue& y, const SVGLengthValue& width, const SVGLengthValue& height);
    static FloatPoint resolvePoint(const SVGElement*, SVGUnitTypes::SVGUnitType, const SVGLengthValue& x, const SVGLengthValue& y);
    static float resolveLength(const SVGElement*, SVGUnitTypes::SVGUnitType, const SVGLengthValue&);

    float valueForLength(const Length&, SVGLengthMode = SVGLengthMode::Other);
    ExceptionOr<float> convertValueToUserUnits(float, SVGLengthType, SVGLengthMode) const;
    ExceptionOr<float> convertValueFromUserUnits(float, SVGLengthType, SVGLengthMode) const;

    std::optional<FloatSize> viewportSize() const;

private:
    ExceptionOr<float> convertValueFromUserUnitsToPercentage(float value, SVGLengthMode) const;
    ExceptionOr<float> convertValueFromPercentageToUserUnits(float value, SVGLengthMode) const;
    static float convertValueFromPercentageToUserUnits(float value, SVGLengthMode, FloatSize);

    ExceptionOr<float> convertValueFromUserUnitsToEMS(float) const;
    ExceptionOr<float> convertValueFromEMSToUserUnits(float) const;

    ExceptionOr<float> convertValueFromUserUnitsToEXS(float) const;
    ExceptionOr<float> convertValueFromEXSToUserUnits(float) const;

    ExceptionOr<float> convertValueFromUserUnitsToLh(float) const;
    ExceptionOr<float> convertValueFromLhToUserUnits(float) const;

    ExceptionOr<float> convertValueFromUserUnitsToCh(float) const;
    ExceptionOr<float> convertValueFromChToUserUnits(float) const;

    std::optional<FloatSize> computeViewportSize() const;

    RefPtr<const SVGElement> protectedContext() const;

    WeakPtr<const SVGElement, WeakPtrImplWithEventTargetData> m_context;
    mutable std::optional<FloatSize> m_viewportSize;
};

} // namespace WebCore
