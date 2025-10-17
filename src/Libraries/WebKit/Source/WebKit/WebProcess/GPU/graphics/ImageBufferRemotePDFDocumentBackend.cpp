/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#include "ImageBufferRemotePDFDocumentBackend.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ImageBufferRemotePDFDocumentBackend);

unsigned ImageBufferRemotePDFDocumentBackend::calculateBytesPerRow(const IntSize& backendSize)
{
    ASSERT(!backendSize.isEmpty());
    return CheckedUint32(backendSize.width()) * 4;
}

size_t ImageBufferRemotePDFDocumentBackend::calculateMemoryCost(const Parameters& parameters)
{
    // FIXME: This is fairly meaningless, because we don't actually have a bitmap, and
    // should really be based on the PDF document size.
    return ImageBufferBackend::calculateMemoryCost(parameters.backendSize, calculateBytesPerRow(parameters.backendSize));
}

std::unique_ptr<ImageBufferRemotePDFDocumentBackend> ImageBufferRemotePDFDocumentBackend::create(const Parameters& parameters)
{
    return std::unique_ptr<ImageBufferRemotePDFDocumentBackend> { new ImageBufferRemotePDFDocumentBackend { parameters } };
}

ImageBufferRemotePDFDocumentBackend::~ImageBufferRemotePDFDocumentBackend() = default;

String ImageBufferRemotePDFDocumentBackend::debugDescription() const
{
    TextStream stream;
    stream << "ImageBufferRemotePDFDocumentBackend " << this;
    return stream.release();
}

} // namespace WebCore
