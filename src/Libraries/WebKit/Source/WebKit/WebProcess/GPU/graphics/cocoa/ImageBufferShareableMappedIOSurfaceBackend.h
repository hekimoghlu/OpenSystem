/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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

#if ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)

#include "ImageBufferBackendHandleSharing.h"
#include <WebCore/ImageBufferIOSurfaceBackend.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class ProcessIdentity;
}

namespace WebKit {

class ImageBufferShareableMappedIOSurfaceBackend final : public WebCore::ImageBufferIOSurfaceBackend, public ImageBufferBackendHandleSharing {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferShareableMappedIOSurfaceBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferShareableMappedIOSurfaceBackend);
public:
    static std::unique_ptr<ImageBufferShareableMappedIOSurfaceBackend> create(const Parameters&, const WebCore::ImageBufferCreationContext&);
    static std::unique_ptr<ImageBufferShareableMappedIOSurfaceBackend> create(const Parameters&, ImageBufferBackendHandle);

    using WebCore::ImageBufferIOSurfaceBackend::ImageBufferIOSurfaceBackend;

    std::optional<ImageBufferBackendHandle> createBackendHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) const final;

private:
    // ImageBufferBackendSharing
    ImageBufferBackendSharing* toBackendSharing() final { return this; }

    String debugDescription() const final;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && HAVE(IOSURFACE)
