/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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

#include "ImageBufferBackend.h"
#include "NullGraphicsContext.h"
#include <memory.h>

namespace WebCore {

// Used for ImageBuffers that return NullGraphicsContext as the ImageBuffer::context().
// Solves the problem of holding NullGraphicsContext similarly to holding other
// GraphicsContext instances, via a ImageBuffer reference.
class WEBCORE_EXPORT NullImageBufferBackend : public ImageBufferBackend {
public:
    static std::unique_ptr<NullImageBufferBackend> create(const Parameters&, const ImageBufferCreationContext&);
    ~NullImageBufferBackend();
    static size_t calculateMemoryCost(const Parameters&) { return 0; }

    NullGraphicsContext& context() override;
    RefPtr<NativeImage> copyNativeImage() override;
    RefPtr<NativeImage> createNativeImageReference() override;
    void getPixelBuffer(const IntRect&, PixelBuffer&) override;
    void putPixelBuffer(const PixelBuffer&, const IntRect&, const IntPoint&, AlphaPremultiplication) override;
    bool canMapBackingStore() const override;
    String debugDescription() const override;

protected:
    using ImageBufferBackend::ImageBufferBackend;
    unsigned bytesPerRow() const override;

    NullGraphicsContext m_context;
};

} // namespace WebCore
