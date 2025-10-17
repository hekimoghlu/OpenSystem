/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

#include "APIObject.h"
#include "ImageOptions.h"
#include <WebCore/ShareableBitmap.h>
#include <wtf/Ref.h>

namespace WebCore {
class ChromeClient;
class GraphicsContext;
class ImageBuffer;
class IntSize;
class NativeImage;
struct ImageBufferParameters;
}

namespace WebKit {

// WebImage - An image type suitable for vending to an API.

class WebImage final : public API::ObjectImpl<API::Object::Type::Image> {
public:
    using ParametersAndHandle = std::pair<WebCore::ImageBufferParameters, WebCore::ShareableBitmap::Handle>;

    static Ref<WebImage> create(const WebCore::IntSize&, ImageOptions, const WebCore::DestinationColorSpace&, WebCore::ChromeClient* = nullptr);
    static Ref<WebImage> create(std::optional<ParametersAndHandle>&&);
    static Ref<WebImage> create(Ref<WebCore::ImageBuffer>&&);
    static Ref<WebImage> createEmpty();

    virtual ~WebImage();

    WebCore::IntSize size() const;
    const WebCore::ImageBufferParameters* parameters() const;
    std::optional<ParametersAndHandle> parametersAndHandle() const;
    bool isEmpty() const { return !m_buffer; }

    WebCore::GraphicsContext* context() const;

    RefPtr<WebCore::NativeImage> copyNativeImage(WebCore::BackingStoreCopy = WebCore::CopyBackingStore) const;
    RefPtr<WebCore::ShareableBitmap> bitmap() const;
#if USE(CAIRO)
    RefPtr<cairo_surface_t> createCairoSurface();
#endif

    std::optional<WebCore::ShareableBitmap::Handle> createHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) const;

private:
    WebImage(RefPtr<WebCore::ImageBuffer>&&);

    RefPtr<WebCore::ImageBuffer> m_buffer;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::WebImage)
static bool isType(const API::Object& object) { return object.type() == API::Object::Type::Image; }
SPECIALIZE_TYPE_TRAITS_END()
