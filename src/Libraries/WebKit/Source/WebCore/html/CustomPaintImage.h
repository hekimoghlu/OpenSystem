/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#include "GeneratedImage.h"
#include "PaintWorkletGlobalScope.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class ImageBuffer;
class RenderElement;

class CustomPaintImage final : public GeneratedImage {
public:
    static Ref<CustomPaintImage> create(PaintDefinition& definition, const FloatSize& size, const RenderElement& element, const Vector<String>& arguments)
    {
        return adoptRef(*new CustomPaintImage(definition, size, element, arguments));
    }

    virtual ~CustomPaintImage();
    bool isCustomPaintImage() const override { return true; }

private:
    CustomPaintImage(PaintDefinition&, const FloatSize&, const RenderElement&, const Vector<String>& arguments);

    ImageDrawResult doCustomPaint(GraphicsContext&, const FloatSize&);

    ImageDrawResult draw(GraphicsContext&, const FloatRect& dstRect, const FloatRect& srcRect, ImagePaintingOptions = { }) final;
    void drawPattern(GraphicsContext&, const FloatRect& destRect, const FloatRect& srcRect, const AffineTransform& patternTransform, const FloatPoint& phase, const FloatSize& spacing, ImagePaintingOptions = { }) final;

    WeakPtr<PaintDefinition> m_paintDefinition;
    Vector<AtomString> m_inputProperties;
    SingleThreadWeakPtr<const RenderElement> m_element;
    Vector<String> m_arguments;
};

}

SPECIALIZE_TYPE_TRAITS_IMAGE(CustomPaintImage)
