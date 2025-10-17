/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#include "MediaStrategy.h"

#include "MediaPlayer.h"
#if ENABLE(MEDIA_SOURCE)
#include "DeprecatedGlobalSettings.h"
#include "MockMediaPlayerMediaSource.h"
#endif

namespace WebCore {

MediaStrategy::MediaStrategy() = default;

MediaStrategy::~MediaStrategy() = default;

std::unique_ptr<NowPlayingManager> MediaStrategy::createNowPlayingManager() const
{
    return makeUnique<NowPlayingManager>();
}

void MediaStrategy::resetMediaEngines()
{
#if ENABLE(VIDEO)
    MediaPlayer::resetMediaEngines();
#endif
    m_mockMediaSourceEnabled = false;
}

bool MediaStrategy::hasThreadSafeMediaSourceSupport() const
{
    return false;
}

#if ENABLE(MEDIA_SOURCE)
void MediaStrategy::enableMockMediaSource()
{
#if USE(AVFOUNDATION)
    WebCore::DeprecatedGlobalSettings::setAVFoundationEnabled(false);
#endif
#if USE(GSTREAMER)
    WebCore::DeprecatedGlobalSettings::setGStreamerEnabled(false);
#endif
    addMockMediaSourceEngine();
}

bool MediaStrategy::mockMediaSourceEnabled() const
{
    return m_mockMediaSourceEnabled;
}

void MediaStrategy::addMockMediaSourceEngine()
{
    MediaPlayerFactorySupport::callRegisterMediaEngine(MockMediaPlayerMediaSource::registerMediaEngine);
}
#endif

#if PLATFORM(COCOA) && ENABLE(MEDIA_RECORDER)
std::unique_ptr<MediaRecorderPrivateWriter> MediaStrategy::createMediaRecorderPrivateWriter(MediaRecorderContainerType, MediaRecorderPrivateWriterListener&) const
{
    return nullptr;
}

#endif

}
