/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include "ImageBufferCGPDFDocumentBackend.h"

#if USE(CG)

#include "GraphicsContext.h"
#include "GraphicsContextCG.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferCGPDFDocumentBackend);

size_t ImageBufferCGPDFDocumentBackend::calculateMemoryCost(const Parameters& parameters)
{
    // FIXME: This is fairly meaningless, because we don't actually have a bitmap, and
    // should really be based on the PDF document size.
    return ImageBufferBackend::calculateMemoryCost(parameters.backendSize, calculateBytesPerRow(parameters.backendSize));
}

std::unique_ptr<ImageBufferCGPDFDocumentBackend> ImageBufferCGPDFDocumentBackend::create(const Parameters& parameters, const ImageBufferCreationContext&)
{
    auto data = adoptCF(CFDataCreateMutable(kCFAllocatorDefault, 0));

    auto dataConsumer = adoptCF(CGDataConsumerCreateWithCFData(data.get()));

    auto backendSize = parameters.backendSize;
    auto mediaBox = CGRectMake(0, 0, backendSize.width(), backendSize.height());

    auto pdfContext = adoptCF(CGPDFContextCreate(dataConsumer.get(), &mediaBox, nullptr));
    auto context = makeUnique<GraphicsContextCG>(pdfContext.get());

    return std::unique_ptr<ImageBufferCGPDFDocumentBackend>(new ImageBufferCGPDFDocumentBackend(parameters, WTFMove(data), WTFMove(context)));
}

ImageBufferCGPDFDocumentBackend::ImageBufferCGPDFDocumentBackend(const Parameters& parameters, RetainPtr<CFDataRef>&& data, std::unique_ptr<GraphicsContextCG>&& context)
    : ImageBufferCGBackend(parameters, WTFMove(context))
    , m_data(WTFMove(data))
{
    ASSERT(m_data);
    ASSERT(m_context);
}

ImageBufferCGPDFDocumentBackend::~ImageBufferCGPDFDocumentBackend() = default;

GraphicsContext& ImageBufferCGPDFDocumentBackend::context()
{
    return *m_context;
}

RefPtr<SharedBuffer> ImageBufferCGPDFDocumentBackend::sinkIntoPDFDocument()
{
    CGPDFContextClose(m_context->platformContext());
    return SharedBuffer::create(m_data.get());
}

String ImageBufferCGPDFDocumentBackend::debugDescription() const
{
    TextStream stream;
    stream << "ImageBufferCGPDFDocumentBackend " << this;
    return stream.release();
}

} // namespace WebCore

#endif // USE(CG)
