/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#import "PlaybackSessionInterfaceMac.h"

#if PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)

#import "IntRect.h"
#import "Logging.h"
#import "MediaSelectionOption.h"
#import "PlaybackSessionModel.h"
#import "TimeRanges.h"
#import "WebPlaybackControlsManager.h"
#import <AVFoundation/AVTime.h>
#import <pal/avfoundation/MediaTimeAVFoundation.h>
#import <pal/spi/cocoa/AVKitSPI.h>
#import <wtf/TZoneMallocInlines.h>

#import <pal/cf/CoreMediaSoftLink.h>

SOFTLINK_AVKIT_FRAMEWORK()
SOFT_LINK_CLASS_OPTIONAL(AVKit, AVValueTiming)

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PlaybackSessionInterfaceMac);

Ref<PlaybackSessionInterfaceMac> PlaybackSessionInterfaceMac::create(PlaybackSessionModel& model)
{
    auto interface = adoptRef(*new PlaybackSessionInterfaceMac(model));
    model.addClient(interface);
    return interface;
}

PlaybackSessionInterfaceMac::PlaybackSessionInterfaceMac(PlaybackSessionModel& model)
    : m_playbackSessionModel(model)
{
}

PlaybackSessionInterfaceMac::~PlaybackSessionInterfaceMac()
{
    invalidate();
}

PlaybackSessionModel* PlaybackSessionInterfaceMac::playbackSessionModel() const
{
    return m_playbackSessionModel.get();
}

bool PlaybackSessionInterfaceMac::isInWindowFullscreenActive() const
{
    return m_playbackSessionModel && m_playbackSessionModel->isInWindowFullscreenActive();
}

void PlaybackSessionInterfaceMac::enterInWindowFullscreen()
{
    if (m_playbackSessionModel)
        m_playbackSessionModel->enterInWindowFullscreen();
}

void PlaybackSessionInterfaceMac::exitInWindowFullscreen()
{
    if (m_playbackSessionModel)
        m_playbackSessionModel->exitInWindowFullscreen();
}

void PlaybackSessionInterfaceMac::durationChanged(double duration)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    WebPlaybackControlsManager* controlsManager = playBackControlsManager();

    controlsManager.contentDuration = duration;

    // FIXME: We take this as an indication that playback is ready, but that is not necessarily true.
    controlsManager.hasEnabledAudio = YES;
    controlsManager.hasEnabledVideo = YES;
#else
    UNUSED_PARAM(duration);
#endif
}

void PlaybackSessionInterfaceMac::currentTimeChanged(double currentTime, double anchorTime)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    WebPlaybackControlsManager* controlsManager = playBackControlsManager();
    updatePlaybackControlsManagerTiming(currentTime, anchorTime, controlsManager.rate, controlsManager.playing);
#else
    UNUSED_PARAM(currentTime);
    UNUSED_PARAM(anchorTime);
#endif
}

void PlaybackSessionInterfaceMac::rateChanged(OptionSet<PlaybackSessionModel::PlaybackState> playbackState, double playbackRate, double defaultPlaybackRate)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    auto isPlaying = playbackState.contains(PlaybackSessionModel::PlaybackState::Playing);
    WebPlaybackControlsManager* controlsManager = playBackControlsManager();
    [controlsManager setDefaultPlaybackRate:defaultPlaybackRate fromJavaScript:YES];
    [controlsManager setRate:isPlaying ? playbackRate : 0. fromJavaScript:YES];
    [controlsManager setPlaying:isPlaying];
    updatePlaybackControlsManagerTiming(m_playbackSessionModel ? m_playbackSessionModel->currentTime() : 0, [[NSProcessInfo processInfo] systemUptime], playbackRate, isPlaying);
#else
    UNUSED_PARAM(isPlaying);
    UNUSED_PARAM(playbackRate);
#endif
}

void PlaybackSessionInterfaceMac::willBeginScrubbing()
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    updatePlaybackControlsManagerTiming(m_playbackSessionModel ? m_playbackSessionModel->currentTime() : 0, [[NSProcessInfo processInfo] systemUptime], 0, false);
#endif
}

void PlaybackSessionInterfaceMac::beginScrubbing()
{
    willBeginScrubbing();
    if (auto* model = playbackSessionModel())
        model->beginScrubbing();
}

void PlaybackSessionInterfaceMac::endScrubbing()
{
    if (auto* model = playbackSessionModel())
        model->endScrubbing();
}

#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
static RetainPtr<NSMutableArray> timeRangesToArray(const TimeRanges& timeRanges)
{
    RetainPtr<NSMutableArray> rangeArray = adoptNS([[NSMutableArray alloc] init]);

    for (unsigned i = 0; i < timeRanges.length(); i++) {
        const PlatformTimeRanges& ranges = timeRanges.ranges();
        CMTimeRange range = PAL::CMTimeRangeMake(PAL::toCMTime(ranges.start(i)), PAL::toCMTime(ranges.end(i)));
        [rangeArray addObject:[NSValue valueWithCMTimeRange:range]];
    }

    return rangeArray;
}
#endif

void PlaybackSessionInterfaceMac::seekableRangesChanged(const TimeRanges& timeRanges, double, double)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    [playBackControlsManager() setSeekableTimeRanges:timeRangesToArray(timeRanges).get()];
#else
    UNUSED_PARAM(timeRanges);
#endif
}

void PlaybackSessionInterfaceMac::audioMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& options, uint64_t selectedIndex)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    [playBackControlsManager() setAudioMediaSelectionOptions:options withSelectedIndex:static_cast<NSUInteger>(selectedIndex)];
#else
    UNUSED_PARAM(options);
    UNUSED_PARAM(selectedIndex);
#endif
}

void PlaybackSessionInterfaceMac::legibleMediaSelectionOptionsChanged(const Vector<MediaSelectionOption>& options, uint64_t selectedIndex)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    [playBackControlsManager() setLegibleMediaSelectionOptions:options withSelectedIndex:static_cast<NSUInteger>(selectedIndex)];
#else
    UNUSED_PARAM(options);
    UNUSED_PARAM(selectedIndex);
#endif
}

void PlaybackSessionInterfaceMac::audioMediaSelectionIndexChanged(uint64_t selectedIndex)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    [playBackControlsManager() setAudioMediaSelectionIndex:selectedIndex];
#else
    UNUSED_PARAM(selectedIndex);
#endif
}

void PlaybackSessionInterfaceMac::legibleMediaSelectionIndexChanged(uint64_t selectedIndex)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    [playBackControlsManager() setLegibleMediaSelectionIndex:selectedIndex];
#else
    UNUSED_PARAM(selectedIndex);
#endif
}

void PlaybackSessionInterfaceMac::isPictureInPictureSupportedChanged(bool)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    updatePlaybackControlsManagerCanTogglePictureInPicture();
#endif
}

void PlaybackSessionInterfaceMac::externalPlaybackChanged(bool, PlaybackSessionModel::ExternalPlaybackTargetType, const String&)
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    updatePlaybackControlsManagerCanTogglePictureInPicture();
#endif
}

void PlaybackSessionInterfaceMac::invalidate()
{
    if (!m_playbackSessionModel)
        return;

    m_playbackSessionModel->removeClient(*this);
    m_playbackSessionModel = nullptr;
}

void PlaybackSessionInterfaceMac::ensureControlsManager()
{
#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)
    playBackControlsManager();
#endif
}

#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)

WebPlaybackControlsManager *PlaybackSessionInterfaceMac::playBackControlsManager()
{
    return m_playbackControlsManager.getAutoreleased();
}

void PlaybackSessionInterfaceMac::setPlayBackControlsManager(WebPlaybackControlsManager *manager)
{
    m_playbackControlsManager = manager;

    if (!manager || !m_playbackSessionModel)
        return;

    NSTimeInterval anchorTimeStamp = ![manager rate] ? NAN : [[NSProcessInfo processInfo] systemUptime];
    manager.timing = [getAVValueTimingClass() valueTimingWithAnchorValue:m_playbackSessionModel->currentTime() anchorTimeStamp:anchorTimeStamp rate:0];
    double duration = m_playbackSessionModel->duration();
    manager.contentDuration = duration;
    manager.hasEnabledAudio = duration > 0;
    manager.hasEnabledVideo = duration > 0;
    manager.defaultPlaybackRate = m_playbackSessionModel->defaultPlaybackRate();
    manager.rate = m_playbackSessionModel->isPlaying() ? m_playbackSessionModel->playbackRate() : 0.;
    manager.seekableTimeRanges = timeRangesToArray(m_playbackSessionModel->seekableRanges()).get();
    manager.canTogglePlayback = YES;
    manager.playing = m_playbackSessionModel->isPlaying();
    [manager setAudioMediaSelectionOptions:m_playbackSessionModel->audioMediaSelectionOptions() withSelectedIndex:static_cast<NSUInteger>(m_playbackSessionModel->audioMediaSelectedIndex())];
    [manager setLegibleMediaSelectionOptions:m_playbackSessionModel->legibleMediaSelectionOptions() withSelectedIndex:static_cast<NSUInteger>(m_playbackSessionModel->legibleMediaSelectedIndex())];

    updatePlaybackControlsManagerCanTogglePictureInPicture();
}

void PlaybackSessionInterfaceMac::updatePlaybackControlsManagerCanTogglePictureInPicture()
{
    PlaybackSessionModel* model = playbackSessionModel();
    if (!model) {
        [playBackControlsManager() setCanTogglePictureInPicture:NO];
        return;
    }

    [playBackControlsManager() setCanTogglePictureInPicture:model->isPictureInPictureSupported() && !model->externalPlaybackEnabled()];
}

void PlaybackSessionInterfaceMac::updatePlaybackControlsManagerTiming(double currentTime, double anchorTime, double playbackRate, bool isPlaying)
{
    WebPlaybackControlsManager *manager = playBackControlsManager();
    if (!manager)
        return;

    PlaybackSessionModel *model = playbackSessionModel();
    if (!model)
        return;

    double effectiveAnchorTime = playbackRate ? anchorTime : NAN;
    double effectivePlaybackRate = playbackRate;
    if (!isPlaying
        || model->isScrubbing()
        || (manager.rate > 0 && model->playbackStartedTime() >= currentTime)
        || (manager.rate < 0 && model->playbackStartedTime() <= currentTime))
        effectivePlaybackRate = 0;

    manager.timing = [getAVValueTimingClass() valueTimingWithAnchorValue:currentTime anchorTimeStamp:effectiveAnchorTime rate:effectivePlaybackRate];
}

uint32_t PlaybackSessionInterfaceMac::checkedPtrCount() const
{
    return CanMakeCheckedPtr::checkedPtrCount();
}

uint32_t PlaybackSessionInterfaceMac::checkedPtrCountWithoutThreadCheck() const
{
    return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck();
}

void PlaybackSessionInterfaceMac::incrementCheckedPtrCount() const
{
    CanMakeCheckedPtr::incrementCheckedPtrCount();
}

void PlaybackSessionInterfaceMac::decrementCheckedPtrCount() const
{
    CanMakeCheckedPtr::decrementCheckedPtrCount();
}

#endif // ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)

#if !RELEASE_LOG_DISABLED
uint64_t PlaybackSessionInterfaceMac::logIdentifier() const
{
    return m_playbackSessionModel ? m_playbackSessionModel->logIdentifier() : 0;
}

const Logger* PlaybackSessionInterfaceMac::loggerPtr() const
{
    return m_playbackSessionModel ? m_playbackSessionModel->loggerPtr() : nullptr;
}

WTFLogChannel& PlaybackSessionInterfaceMac::logChannel() const
{
    return LogMedia;
}

#endif

}

#endif // PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)
