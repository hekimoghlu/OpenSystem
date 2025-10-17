/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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

#import <utility>
#import <wtf/CompletionHandler.h>
#import <wtf/FastMalloc.h>
#import <wtf/Ref.h>
#import <wtf/WeakPtr.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

struct WGPUXRSubImageImpl {
};

namespace WebGPU {

class CommandEncoder;
class Device;
class Texture;

class XRSubImage : public RefCountedAndCanMakeWeakPtr<XRSubImage>, public WGPUXRSubImageImpl {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<XRSubImage> create(Device& device)
    {
        return adoptRef(*new XRSubImage(true, device));
    }
    static Ref<XRSubImage> createInvalid(Device& device)
    {
        return adoptRef(*new XRSubImage(device));
    }

    ~XRSubImage();

    void setLabel(String&&);

    bool isValid() const;
    void update(id<MTLTexture> colorTexture, id<MTLTexture> depthTexture, size_t currentTextureIndex, const std::pair<id<MTLSharedEvent>, uint64_t>&);
    Texture* colorTexture();
    Texture* depthTexture();

private:
    XRSubImage(bool, Device&);
    XRSubImage(Device&);

    HashMap<uint64_t, RefPtr<Texture>, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_colorTextures;
    HashMap<uint64_t, RefPtr<Texture>, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>> m_depthTextures;
    uint64_t m_currentTextureIndex { 0 };

    ThreadSafeWeakPtr<Device> m_device;
};

} // namespace WebGPU
