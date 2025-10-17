/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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

#if ENABLE(GPU_PROCESS)

#include <WebCore/NativeImage.h>
#include <WebCore/ShareableBitmap.h>

namespace WebKit {
class RemoteResourceCacheProxy;

class RemoteNativeImageBackendProxy final : public WebCore::NativeImageBackend {
public:
    static std::unique_ptr<RemoteNativeImageBackendProxy> create(WebCore::NativeImage&, const WebCore::DestinationColorSpace&);
    ~RemoteNativeImageBackendProxy() final;

    const WebCore::PlatformImagePtr& platformImage() const final;
    WebCore::IntSize size() const final;
    bool hasAlpha() const final;
    WebCore::DestinationColorSpace colorSpace() const final;
    WebCore::Headroom headroom() const final;
    bool isRemoteNativeImageBackendProxy() const final;

    std::optional<WebCore::ShareableBitmap::Handle> createHandle();
private:
    RemoteNativeImageBackendProxy(Ref<WebCore::ShareableBitmap>, WebCore::PlatformImagePtr);

    Ref<WebCore::ShareableBitmap> m_bitmap;
    WebCore::PlatformImageNativeImageBackend m_platformBackend;
};

}
SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteNativeImageBackendProxy)
    static bool isType(const WebCore::NativeImageBackend& backend) { return backend.isRemoteNativeImageBackendProxy(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif
