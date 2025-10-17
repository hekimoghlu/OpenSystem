/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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
#include "ShareablePixelBuffer.h"

#include <WebCore/SharedMemory.h>

namespace WebKit {
using namespace WebCore;

RefPtr<ShareablePixelBuffer> ShareablePixelBuffer::tryCreate(const PixelBufferFormat& format, const IntSize& size)
{
    ASSERT(supportedPixelFormat(format.pixelFormat));

    auto bufferSize = computeBufferSize(format.pixelFormat, size);
    if (bufferSize.hasOverflowed())
        return nullptr;
    RefPtr<SharedMemory> sharedMemory = SharedMemory::allocate(bufferSize);
    if (!sharedMemory)
        return nullptr;

    return adoptRef(new ShareablePixelBuffer(format, size, sharedMemory.releaseNonNull()));
}

ShareablePixelBuffer::ShareablePixelBuffer(const PixelBufferFormat& format, const IntSize& size, Ref<SharedMemory>&& data)
    : PixelBuffer(format, size, data->mutableSpan())
    , m_data(WTFMove(data))
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(m_size.area() * 4 <= bytes().size());
}

RefPtr<PixelBuffer> ShareablePixelBuffer::createScratchPixelBuffer(const IntSize& size) const
{
    return ShareablePixelBuffer::tryCreate(m_format, size);
}

Ref<WebCore::SharedMemory> ShareablePixelBuffer::protectedData() const
{
    return m_data;
}

} // namespace WebKit
