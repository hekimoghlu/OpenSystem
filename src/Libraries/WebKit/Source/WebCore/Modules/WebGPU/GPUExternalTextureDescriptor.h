/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

#include "GPUObjectDescriptorBase.h"
#include "GPUPredefinedColorSpace.h"
#include "HTMLVideoElement.h"
#include "WebCodecsVideoFrame.h"
#include "WebGPUExternalTextureDescriptor.h"
#include <wtf/RefPtr.h>

typedef struct __CVBuffer* CVPixelBufferRef;

namespace WebCore {

class HTMLVideoElement;
#if ENABLE(WEB_CODECS)
using GPUVideoSource = std::variant<RefPtr<HTMLVideoElement>, RefPtr<WebCodecsVideoFrame>>;
#else
using GPUVideoSource = RefPtr<HTMLVideoElement>;
#endif

struct GPUExternalTextureDescriptor : public GPUObjectDescriptorBase {

#if ENABLE(VIDEO)
    static WebGPU::VideoSourceIdentifier mediaIdentifierForSource(const GPUVideoSource& videoSource)
    {
#if ENABLE(WEB_CODECS)
        return WTF::switchOn(videoSource, [&](const RefPtr<HTMLVideoElement> videoElement) -> WebGPU::VideoSourceIdentifier {
            if (auto playerIdentifier = videoElement->playerIdentifier())
                return playerIdentifier;
            RefPtr<WebCore::VideoFrame> result;
            if (videoElement->player())
                result = videoElement->protectedPlayer()->videoFrameForCurrentTime();
            return result;
        }
        , [&](const RefPtr<WebCodecsVideoFrame> videoFrame) -> WebGPU::VideoSourceIdentifier {
            return videoFrame->internalFrame();
        });
#else
        return videoSource->playerIdentifier();
#endif
    }

    std::optional<WebCore::MediaPlayerIdentifier> mediaIdentifier() const
    {
#if ENABLE(WEB_CODECS)
        return WTF::switchOn(source, [&](const RefPtr<HTMLVideoElement> videoElement) -> std::optional<WebCore::MediaPlayerIdentifier> {
            return videoElement->playerIdentifier();
        }
        , [&](const RefPtr<WebCodecsVideoFrame>) -> std::optional<WebCore::MediaPlayerIdentifier> {
            return std::nullopt;
        });
#else
        return source->playerIdentifier();
#endif
    }
#endif

    WebGPU::ExternalTextureDescriptor convertToBacking() const
    {
        return {
            { label },
#if ENABLE(VIDEO)
            mediaIdentifierForSource(source),
#else
            { },
#endif
            WebCore::convertToBacking(colorSpace),
        };
    }

#if ENABLE(VIDEO)
    GPUVideoSource source;
#endif
    GPUPredefinedColorSpace colorSpace { GPUPredefinedColorSpace::SRGB };
};

}
