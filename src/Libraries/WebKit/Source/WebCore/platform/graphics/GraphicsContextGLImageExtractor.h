/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

#if ENABLE(WEBGL)

#include "GraphicsContextGL.h"
#include <wtf/MallocSpan.h>

namespace WebCore {

class GraphicsContextGLImageExtractor {
public:
    using DOMSource = GraphicsContextGL::DOMSource;
    using DataFormat = GraphicsContextGL::DataFormat;
    using AlphaOp = GraphicsContextGL::AlphaOp;
    GraphicsContextGLImageExtractor(Image&, DOMSource, bool premultiplyAlpha, bool ignoreGammaAndColorProfile, bool ignoreNativeImageAlphaPremultiplication);

    // Each platform must provide an implementation of this method to deallocate or release resources
    // associated with the image if needed.
    ~GraphicsContextGLImageExtractor();

    bool extractSucceeded() { return m_extractSucceeded; }
    std::span<const uint8_t> imagePixelData() { return m_imagePixelData; }
    unsigned imageWidth() { return m_imageWidth; }
    unsigned imageHeight() { return m_imageHeight; }
    DataFormat imageSourceFormat() { return m_imageSourceFormat; }
    AlphaOp imageAlphaOp() { return m_alphaOp; }
    unsigned imageSourceUnpackAlignment() { return m_imageSourceUnpackAlignment; }
    DOMSource imageHtmlDomSource() { return m_imageHtmlDomSource; }
private:
    // Each platform must provide an implementation of this method.
    // Extracts the image and keeps track of its status, such as width, height, Source Alignment, format and AlphaOp etc,
    // needs to lock the resources or relevant data if needed and returns true upon success
    bool extractImage(bool premultiplyAlpha, bool ignoreGammaAndColorProfile, bool ignoreNativeImageAlphaPremultiplication);

#if USE(CAIRO)
    RefPtr<cairo_surface_t> m_imageSurface;
#elif USE(CG)
    RetainPtr<CFDataRef> m_pixelData;
    MallocSpan<uint8_t> m_formalizedRGBA8Data;
#elif USE(SKIA)
    sk_sp<SkData> m_pixelData;
    sk_sp<SkImage> m_skImage;
#endif
    Ref<Image> m_image;
    DOMSource m_imageHtmlDomSource;
    bool m_extractSucceeded;
    std::span<const uint8_t> m_imagePixelData;
    unsigned m_imageWidth;
    unsigned m_imageHeight;
    DataFormat m_imageSourceFormat;
    AlphaOp m_alphaOp;
    unsigned m_imageSourceUnpackAlignment;
};

}

#endif
