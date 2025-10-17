/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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

#include "FloatRect.h"
#include "IntSize.h"
#include "NativeImage.h"
#include "PlatformLayer.h"
#include "VideoLayerManager.h"
#include <wtf/Function.h>
#include <wtf/LoggerHelper.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebVideoContainerLayer;

namespace WebCore {

class VideoLayerManagerObjC final
    : public VideoLayerManager
#if !RELEASE_LOG_DISABLED
    , public LoggerHelper
#endif
{
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(VideoLayerManagerObjC, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(VideoLayerManagerObjC);

public:
#if !RELEASE_LOG_DISABLED
    WEBCORE_EXPORT VideoLayerManagerObjC(const Logger&, uint64_t);
#else
    VideoLayerManagerObjC() = default;
#endif

    WEBCORE_EXPORT ~VideoLayerManagerObjC();

    WEBCORE_EXPORT PlatformLayer* videoInlineLayer() const final;

    WEBCORE_EXPORT void setVideoLayer(PlatformLayer*, FloatSize) final;
    WEBCORE_EXPORT void didDestroyVideoLayer() final;

#if ENABLE(VIDEO_PRESENTATION_MODE)
    WEBCORE_EXPORT PlatformLayer* videoFullscreenLayer() const final;
    WEBCORE_EXPORT void setVideoFullscreenLayer(PlatformLayer*, Function<void()>&& completionHandler, PlatformImagePtr) final;
    WEBCORE_EXPORT FloatRect videoFullscreenFrame() const final;
    WEBCORE_EXPORT void setVideoFullscreenFrame(FloatRect) final;
    WEBCORE_EXPORT void updateVideoFullscreenInlineImage(PlatformImagePtr) final;
#endif

    WEBCORE_EXPORT void setTextTrackRepresentationLayer(PlatformLayer*) final;
    WEBCORE_EXPORT void syncTextTrackBounds() final;

private:

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "VideoLayerManagerObjC"_s; }
    WTFLogChannel& logChannel() const final;

    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

    RetainPtr<WebVideoContainerLayer> m_videoInlineLayer;
#if ENABLE(VIDEO_PRESENTATION_MODE)
    RetainPtr<PlatformLayer> m_videoFullscreenLayer;
    FloatRect m_videoFullscreenFrame;
#endif
    RetainPtr<PlatformLayer> m_textTrackRepresentationLayer;

    RetainPtr<PlatformLayer> m_videoLayer;
};

}
