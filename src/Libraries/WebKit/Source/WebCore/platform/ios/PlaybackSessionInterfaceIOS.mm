/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#import "config.h"
#import "PlaybackSessionInterfaceIOS.h"

#if PLATFORM(COCOA) && HAVE(AVKIT)

#import "Logging.h"
#import "MediaSelectionOption.h"
#import "PlaybackSessionModel.h"
#import "TimeRanges.h"
#import <AVFoundation/AVTime.h>
#import <wtf/LoggerHelper.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/text/CString.h>
#import <wtf/text/WTFString.h>

#import <pal/cf/CoreMediaSoftLink.h>
#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PlaybackSessionInterfaceIOS);

PlaybackSessionInterfaceIOS::PlaybackSessionInterfaceIOS(PlaybackSessionModel& model)
    : m_playbackSessionModel(&model)
{
    model.addClient(*this);
}

PlaybackSessionInterfaceIOS::~PlaybackSessionInterfaceIOS()
{
    ASSERT(isUIThread());
}

void PlaybackSessionInterfaceIOS::initialize()
{
    auto& model = *m_playbackSessionModel;

    durationChanged(model.duration());
    currentTimeChanged(model.currentTime(), [[NSProcessInfo processInfo] systemUptime]);
    bufferedTimeChanged(model.bufferedTime());
    OptionSet<PlaybackSessionModel::PlaybackState> playbackState;
    if (model.isPlaying())
        playbackState.add(PlaybackSessionModel::PlaybackState::Playing);
    if (model.isStalled())
        playbackState.add(PlaybackSessionModel::PlaybackState::Stalled);
    rateChanged(playbackState, model.playbackRate(), model.defaultPlaybackRate());
    seekableRangesChanged(model.seekableRanges(), model.seekableTimeRangesLastModifiedTime(), model.liveUpdateInterval());
    canPlayFastReverseChanged(model.canPlayFastReverse());
    audioMediaSelectionOptionsChanged(model.audioMediaSelectionOptions(), model.audioMediaSelectedIndex());
    legibleMediaSelectionOptionsChanged(model.legibleMediaSelectionOptions(), model.legibleMediaSelectedIndex());
    externalPlaybackChanged(model.externalPlaybackEnabled(), model.externalPlaybackTargetType(), model.externalPlaybackLocalizedDeviceName());
    wirelessVideoPlaybackDisabledChanged(model.wirelessVideoPlaybackDisabled());
}

void PlaybackSessionInterfaceIOS::invalidate()
{
    if (!m_playbackSessionModel)
        return;
    m_playbackSessionModel->removeClient(*this);
    m_playbackSessionModel = nullptr;
}

PlaybackSessionModel* PlaybackSessionInterfaceIOS::playbackSessionModel() const
{
    return m_playbackSessionModel;
}

void PlaybackSessionInterfaceIOS::modelDestroyed()
{
    ASSERT(isUIThread());
    invalidate();
    ASSERT(!m_playbackSessionModel);
}

std::optional<MediaPlayerIdentifier> PlaybackSessionInterfaceIOS::playerIdentifier() const
{
    return m_playerIdentifier;
}

void PlaybackSessionInterfaceIOS::setPlayerIdentifier(std::optional<MediaPlayerIdentifier> identifier)
{
    m_playerIdentifier = WTFMove(identifier);
}

void PlaybackSessionInterfaceIOS::startObservingNowPlayingMetadata()
{
}

void PlaybackSessionInterfaceIOS::stopObservingNowPlayingMetadata()
{
}

#if !RELEASE_LOG_DISABLED
uint64_t PlaybackSessionInterfaceIOS::logIdentifier() const
{
    return m_playbackSessionModel ? m_playbackSessionModel->logIdentifier() : 0;
}

const Logger* PlaybackSessionInterfaceIOS::loggerPtr() const
{
    return m_playbackSessionModel ? m_playbackSessionModel->loggerPtr() : nullptr;
}

WTFLogChannel& PlaybackSessionInterfaceIOS::logChannel() const
{
    return LogMedia;
}

uint32_t PlaybackSessionInterfaceIOS::checkedPtrCount() const
{
    return CanMakeCheckedPtr::checkedPtrCount();
}

uint32_t PlaybackSessionInterfaceIOS::checkedPtrCountWithoutThreadCheck() const
{
    return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck();
}

void PlaybackSessionInterfaceIOS::incrementCheckedPtrCount() const
{
    CanMakeCheckedPtr::incrementCheckedPtrCount();
}

void PlaybackSessionInterfaceIOS::decrementCheckedPtrCount() const
{
    CanMakeCheckedPtr::decrementCheckedPtrCount();
}

#endif

} // namespace WebCore

#endif // PLATFORM(COCOA) && HAVE(AVKIT)
