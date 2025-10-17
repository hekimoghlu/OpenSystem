/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include "ImageBufferCGBackend.h"

#if USE(CG)

#include "IntRect.h"
#include <CoreGraphics/CoreGraphics.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

class ThreadSafeImageBufferFlusherCG : public ThreadSafeImageBufferFlusher {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ThreadSafeImageBufferFlusherCG);
public:
    ThreadSafeImageBufferFlusherCG(CGContextRef context)
        : m_context(context)
    {
    }

    void flush() override
    {
        CGContextFlush(m_context.get());
    }

private:
    RetainPtr<CGContextRef> m_context;
};

ImageBufferCGBackend::ImageBufferCGBackend(const Parameters& parameters, std::unique_ptr<GraphicsContextCG>&& context)
    : ImageBufferBackend(parameters)
    , m_context(WTFMove(context))
{
}

ImageBufferCGBackend::~ImageBufferCGBackend() = default;

unsigned ImageBufferCGBackend::calculateBytesPerRow(const IntSize& backendSize)
{
    ASSERT(!backendSize.isEmpty());
    return CheckedUint32(backendSize.width()) * 4;
}

std::unique_ptr<ThreadSafeImageBufferFlusher> ImageBufferCGBackend::createFlusher()
{
    return makeUnique<ThreadSafeImageBufferFlusherCG>(context().platformContext());
}

void ImageBufferCGBackend::applyBaseTransform(GraphicsContextCG& context) const
{
    context.applyDeviceScaleFactor(m_parameters.resolutionScale);
    context.setCTM(calculateBaseTransform(m_parameters));
}

String ImageBufferCGBackend::debugDescription() const
{
    TextStream stream;
    stream << "ImageBufferCGBackend " << this;
    return stream.release();
}

} // namespace WebCore

#endif // USE(CG)
