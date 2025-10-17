/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 24, 2024.
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
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/WeakPtr.h>

struct WGPUXRProjectionLayerImpl {
};

namespace WebGPU {

class CommandEncoder;
class Device;

class XRProjectionLayer : public RefCountedAndCanMakeWeakPtr<XRProjectionLayer>, public WGPUXRProjectionLayerImpl {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<XRProjectionLayer> create(Device& device)
    {
        return adoptRef(*new XRProjectionLayer(true, device));
    }
    static Ref<XRProjectionLayer> createInvalid(Device& device)
    {
        return adoptRef(*new XRProjectionLayer(device));
    }

    ~XRProjectionLayer();

    void setLabel(String&&);

    bool isValid() const;
    void startFrame(size_t frameIndex, MachSendRight&& colorBuffer, MachSendRight&& depthBuffer, MachSendRight&& completionSyncEvent, size_t reusableTextureIndex);

    id<MTLTexture> colorTexture() const;
    id<MTLTexture> depthTexture() const;
    const std::pair<id<MTLSharedEvent>, uint64_t>& completionEvent() const;
    size_t reusableTextureIndex() const;

private:
    XRProjectionLayer(bool, Device&);
    XRProjectionLayer(Device&);

    NSMutableDictionary<NSNumber*, id<MTLTexture>>* m_colorTextures { nil };
    NSMutableDictionary<NSNumber*, id<MTLTexture>>* m_depthTextures { nil };
    id<MTLTexture> m_colorTexture { nil };
    id<MTLTexture> m_depthTexture { nil };
    std::pair<id<MTLSharedEvent>, uint64_t> m_sharedEvent;
    size_t m_reusableTextureIndex { 0 };

    Ref<Device> m_device;
};

} // namespace WebGPU
