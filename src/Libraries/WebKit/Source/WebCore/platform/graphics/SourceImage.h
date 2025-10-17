/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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

#include "IntSize.h"
#include "RenderingResourceIdentifier.h"

namespace WebCore {

class ImageBuffer;
class NativeImage;

class WEBCORE_EXPORT SourceImage {
public:
    using ImageVariant = std::variant<
        Ref<NativeImage>,
        Ref<ImageBuffer>,
        RenderingResourceIdentifier
    >;

    SourceImage(ImageVariant&&);

    SourceImage(const SourceImage&);
    SourceImage(SourceImage&&);
    SourceImage& operator=(const SourceImage&);
    SourceImage& operator=(SourceImage&&);
    ~SourceImage();

    bool operator==(const SourceImage&) const;

    NativeImage* nativeImageIfExists() const;
    NativeImage* nativeImage() const;

    ImageBuffer* imageBufferIfExists() const;
    ImageBuffer* imageBuffer() const;

    RenderingResourceIdentifier imageIdentifier() const;
    IntSize size() const;

private:
    ImageVariant m_imageVariant;
    mutable std::optional<ImageVariant> m_transformedImageVariant;
};


} // namespace WebCore
