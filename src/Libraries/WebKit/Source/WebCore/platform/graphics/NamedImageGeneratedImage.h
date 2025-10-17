/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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

#include "FloatSize.h"
#include "GeneratedImage.h"
#include "Image.h"

namespace WebCore {

class NamedImageGeneratedImage final : public GeneratedImage {
public:
    static Ref<NamedImageGeneratedImage> create(String name, const FloatSize& size)
    {
        return adoptRef(*new NamedImageGeneratedImage(name, size));
    }

private:
    ImageDrawResult draw(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions = { }) override;
    void drawPattern(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) override;

    NamedImageGeneratedImage(String name, const FloatSize&);

    bool isNamedImageGeneratedImage() const override { return true; }
    void dump(WTF::TextStream&) const override;

    String m_name;
};

}

SPECIALIZE_TYPE_TRAITS_IMAGE(NamedImageGeneratedImage)
