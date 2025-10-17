/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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

#include "ImageBufferBackendHandle.h"
#include <WebCore/ImageBufferBackend.h>

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
#include <WebCore/DynamicContentScalingDisplayList.h>
#endif

namespace WebKit {

class ImageBufferBackendHandleSharing : public WebCore::ImageBufferBackendSharing {
public:
    virtual std::optional<ImageBufferBackendHandle> createBackendHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) const = 0;
    virtual std::optional<ImageBufferBackendHandle> takeBackendHandle(WebCore::SharedMemory::Protection protection = WebCore::SharedMemory::Protection::ReadWrite) { return createBackendHandle(protection); }

    virtual RefPtr<WebCore::ShareableBitmap> bitmap() const { return nullptr; }

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    virtual std::optional<WebCore::DynamicContentScalingDisplayList> dynamicContentScalingDisplayList() { return std::nullopt; }
#endif

    virtual void setBackendHandle(ImageBufferBackendHandle&&) { }
    virtual void clearBackendHandle() { }

private:
    bool isImageBufferBackendHandleSharing() const final { return true; }
};

} // namespace WebKit

#define SPECIALIZE_TYPE_TRAITS_IMAGE_BUFFER_BACKEND_SHARING(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToValueTypeName) \
    static bool isType(const WebCore::ImageBufferBackendSharing& backendSharing) { return backendSharing.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_IMAGE_BUFFER_BACKEND_SHARING(WebKit::ImageBufferBackendHandleSharing, isImageBufferBackendHandleSharing())
