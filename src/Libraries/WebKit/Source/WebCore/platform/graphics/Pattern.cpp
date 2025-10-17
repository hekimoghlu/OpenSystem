/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#include "config.h"
#include "Pattern.h"

#include "Image.h"
#include "ImageBuffer.h"
#include "NativeImage.h"

namespace WebCore {

Ref<Pattern> Pattern::create(SourceImage&& tileImage, const Parameters& parameters)
{
    return adoptRef(*new Pattern(WTFMove(tileImage), parameters));
}

Pattern::Pattern(SourceImage&& tileImage, const Parameters& parameters)
    : m_tileImage(WTFMove(tileImage))
    , m_parameters(parameters)
{
}

Pattern::~Pattern() = default;

void Pattern::setPatternSpaceTransform(const AffineTransform& patternSpaceTransform)
{
    m_parameters.patternSpaceTransform = patternSpaceTransform;
}

const SourceImage& Pattern::tileImage() const
{
    return m_tileImage;
}

RefPtr<NativeImage> Pattern::tileNativeImage() const
{
    return m_tileImage.nativeImage();
}

RefPtr<ImageBuffer> Pattern::tileImageBuffer() const
{
    return m_tileImage.imageBuffer();
}

void Pattern::setTileImage(SourceImage&& tileImage)
{
    m_tileImage = WTFMove(tileImage);
}

}
