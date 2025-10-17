/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#include "NullImageBufferBackend.h"

#include "PixelBuffer.h"
#include <wtf/text/TextStream.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

std::unique_ptr<NullImageBufferBackend> NullImageBufferBackend::create(const Parameters& parameters, const ImageBufferCreationContext&)
{
    return std::unique_ptr<NullImageBufferBackend> { new NullImageBufferBackend { parameters } };
}

NullImageBufferBackend::~NullImageBufferBackend() = default;

NullGraphicsContext& NullImageBufferBackend::context()
{
    return m_context;
}

RefPtr<NativeImage> NullImageBufferBackend::copyNativeImage()
{
    return nullptr;
}

RefPtr<NativeImage> NullImageBufferBackend::createNativeImageReference()
{
    return nullptr;
}

void NullImageBufferBackend::getPixelBuffer(const IntRect&, PixelBuffer& destination)
{
    destination.zeroFill();
}

void NullImageBufferBackend::putPixelBuffer(const PixelBuffer&, const IntRect&, const IntPoint&, AlphaPremultiplication)
{
}

unsigned NullImageBufferBackend::bytesPerRow() const
{
    return 0;
}

bool NullImageBufferBackend::canMapBackingStore() const
{
    return false;
}

String NullImageBufferBackend::debugDescription() const
{
    TextStream stream;
    stream << "NullImageBufferBackend " << this;
    return stream.release();
}

}
