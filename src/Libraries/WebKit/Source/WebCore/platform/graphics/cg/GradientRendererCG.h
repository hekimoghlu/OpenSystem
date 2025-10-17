/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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

#include "ColorComponents.h"
#include "ColorInterpolationMethod.h"
#include <CoreGraphics/CoreGraphics.h>
#include <wtf/RetainPtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class GradientColorStops;

struct ColorConvertedToInterpolationColorSpaceStop {
    float offset;
    ColorComponents<float, 4> colorComponents;
};

class GradientRendererCG {
public:
    GradientRendererCG(ColorInterpolationMethod, const GradientColorStops&);

    void drawLinearGradient(CGContextRef, CGPoint startPoint, CGPoint endPoint, CGGradientDrawingOptions);
    void drawRadialGradient(CGContextRef, CGPoint startCenter, CGFloat startRadius, CGPoint endCenter, CGFloat endRadius, CGGradientDrawingOptions);
    void drawConicGradient(CGContextRef, CGPoint center, CGFloat angle);

private:
    struct Gradient {
        RetainPtr<CGGradientRef> gradient;
    };

    struct Shading {
        template<typename InterpolationSpace, AlphaPremultiplication> static void shadingFunction(void*, const CGFloat*, CGFloat*);

        class Data : public ThreadSafeRefCounted<Data> {
        public:
            static Ref<Data> create(ColorInterpolationMethod colorInterpolationMethod, Vector<ColorConvertedToInterpolationColorSpaceStop> stops)
            {
                return adoptRef(*new Data(colorInterpolationMethod, WTFMove(stops)));
            }

            ColorInterpolationMethod colorInterpolationMethod() const { return m_colorInterpolationMethod; }
            const Vector<ColorConvertedToInterpolationColorSpaceStop>& stops() const { return m_stops; }

        private:
            Data(ColorInterpolationMethod colorInterpolationMethod, Vector<ColorConvertedToInterpolationColorSpaceStop> stops)
                : m_colorInterpolationMethod { colorInterpolationMethod }
                , m_stops { WTFMove(stops) }
            {
            }

            ColorInterpolationMethod m_colorInterpolationMethod;
            Vector<ColorConvertedToInterpolationColorSpaceStop> m_stops;
        };

        Ref<Data> data;
        RetainPtr<CGFunctionRef> function;
        RetainPtr<CGColorSpaceRef> colorSpace;
    };

    using Strategy = std::variant<Gradient, Shading>;

    Strategy pickStrategy(ColorInterpolationMethod, const GradientColorStops&) const;
    Strategy makeGradient(ColorInterpolationMethod, const GradientColorStops&) const;
    Strategy makeShading(ColorInterpolationMethod, const GradientColorStops&) const;

    Strategy m_strategy;
};

}
