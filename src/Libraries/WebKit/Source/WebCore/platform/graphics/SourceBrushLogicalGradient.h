/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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

#include "Gradient.h"

namespace WebCore {

struct SourceBrushLogicalGradient {
    std::variant<Ref<Gradient>, RenderingResourceIdentifier> gradient;
    AffineTransform spaceTransform;

    std::variant<Ref<Gradient>, RenderingResourceIdentifier> serializableGradient() const;
    friend bool operator==(const SourceBrushLogicalGradient&, const SourceBrushLogicalGradient&);
};

inline std::variant<Ref<Gradient>, RenderingResourceIdentifier> SourceBrushLogicalGradient::serializableGradient() const
{
    return WTF::switchOn(gradient,
        [&] (const Ref<Gradient>& gradient) -> std::variant<Ref<Gradient>, RenderingResourceIdentifier> {
            if (gradient->hasValidRenderingResourceIdentifier())
                return gradient->renderingResourceIdentifier();
            return gradient;
        },
        [&] (RenderingResourceIdentifier renderingResourceIdentifier) -> std::variant<Ref<Gradient>, RenderingResourceIdentifier> {
            return renderingResourceIdentifier;
        }
    );
}

inline bool operator==(const SourceBrushLogicalGradient& a, const SourceBrushLogicalGradient& b)
{
    if (a.spaceTransform != b.spaceTransform)
        return false;

    return WTF::switchOn(a.gradient,
        [&] (const Ref<Gradient>& aGradient) {
            if (auto* bGradient = std::get_if<Ref<Gradient>>(&b.gradient))
                return aGradient.ptr() == bGradient->ptr();
            return false;
        },
        [&] (RenderingResourceIdentifier aRenderingResourceIdentifier) {
            if (auto* bRenderingResourceIdentifier = std::get_if<RenderingResourceIdentifier>(&b.gradient))
                return aRenderingResourceIdentifier == *bRenderingResourceIdentifier;
            return false;
        }
    );
}

} // namespace WebCore
