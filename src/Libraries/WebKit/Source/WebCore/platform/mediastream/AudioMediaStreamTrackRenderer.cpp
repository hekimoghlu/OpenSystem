/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#include "AudioMediaStreamTrackRenderer.h"

#if ENABLE(MEDIA_STREAM)

#include "Logging.h"
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include "AudioMediaStreamTrackRendererCocoa.h"
#endif

#if USE(LIBWEBRTC)
#include "LibWebRTCAudioModule.h"
#endif

namespace WTF {
class MediaTime;
}

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioMediaStreamTrackRenderer);

RefPtr<AudioMediaStreamTrackRenderer> AudioMediaStreamTrackRenderer::create(Init&& init)
{
#if PLATFORM(COCOA)
    return AudioMediaStreamTrackRendererCocoa::create(WTFMove(init));
#else
    UNUSED_PARAM(init);
    return nullptr;
#endif
}

String AudioMediaStreamTrackRenderer::defaultDeviceID()
{
    ASSERT(isMainThread());
    return "default"_s;
}

AudioMediaStreamTrackRenderer::AudioMediaStreamTrackRenderer(Init&& init)
    : m_crashCallback(WTFMove(init.crashCallback))
#if USE(LIBWEBRTC)
    , m_audioModule(WTFMove(init.audioModule))
#endif
#if !RELEASE_LOG_DISABLED
    , m_logger(init.logger)
    , m_logIdentifier(init.logIdentifier)
#endif
{
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& AudioMediaStreamTrackRenderer::logChannel() const
{
    return LogMedia;
}

const Logger& AudioMediaStreamTrackRenderer::logger() const
{
    return m_logger.get();

}

uint64_t AudioMediaStreamTrackRenderer::logIdentifier() const
{
    return m_logIdentifier;
}

ASCIILiteral AudioMediaStreamTrackRenderer::logClassName() const
{
    return "AudioMediaStreamTrackRenderer"_s;
}
#endif

}

#endif // ENABLE(MEDIA_STREAM)
