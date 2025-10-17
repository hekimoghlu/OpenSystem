/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WTF {
class MachSendRight;
}

namespace WebCore {
class NativeImage;
}

namespace WebCore::WebGPU {

struct CanvasConfiguration;
class Texture;

class PresentationContext : public RefCountedAndCanMakeWeakPtr<PresentationContext> {
public:
    virtual ~PresentationContext() = default;

    WARN_UNUSED_RETURN virtual bool configure(const CanvasConfiguration&) = 0;
    virtual void unconfigure() = 0;
    virtual void present(uint32_t frameIndex, bool = false) = 0;

    virtual RefPtr<Texture> getCurrentTexture(uint32_t) = 0;
    virtual RefPtr<WebCore::NativeImage> getMetalTextureAsNativeImage(uint32_t bufferIndex, bool& isIOSurfaceSupportedFormat) = 0;

protected:
    PresentationContext() = default;

private:
    PresentationContext(const PresentationContext&) = delete;
    PresentationContext(PresentationContext&&) = delete;
    PresentationContext& operator=(const PresentationContext&) = delete;
    PresentationContext& operator=(PresentationContext&&) = delete;
};

} // namespace WebCore::WebGPU
