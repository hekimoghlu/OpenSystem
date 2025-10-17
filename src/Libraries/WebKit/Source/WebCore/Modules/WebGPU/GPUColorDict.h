/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

#include "WebGPUColor.h"
#include <variant>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct GPUColorDict {
    WebGPU::ColorDict convertToBacking() const
    {
        return {
            r,
            g,
            b,
            a,
        };
    }

    double r { 0 };
    double g { 0 };
    double b { 0 };
    double a { 0 };
};

using GPUColor = std::variant<Vector<double>, GPUColorDict>;

inline WebGPU::Color convertToBacking(const GPUColor& color)
{
    return WTF::switchOn(color, [](const Vector<double>& vector) -> WebGPU::Color {
        return vector;
    }, [](const GPUColorDict& color) -> WebGPU::Color {
        return color.convertToBacking();
    });
}

}
