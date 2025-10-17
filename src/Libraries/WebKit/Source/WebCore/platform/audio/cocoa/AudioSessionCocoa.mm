/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#import "AudioSessionCocoa.h"

#if USE(AUDIO_SESSION) && PLATFORM(COCOA)

#import "Logging.h"
#import "NotImplemented.h"
#import <AVFoundation/AVAudioSession.h>
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/WorkQueue.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AudioSessionCocoa);

void AudioSessionCocoa::setEligibleForSmartRoutingInternal(bool eligible)
{
#if HAVE(AVAUDIOSESSION_SMARTROUTING)
    if (!AudioSession::shouldManageAudioSessionCategory())
        return;

    static bool supportsEligibleForBT = [PAL::getAVAudioSessionClass() instancesRespondToSelector:@selector(setEligibleForBTSmartRoutingConsideration:error:)]
        && [PAL::getAVAudioSessionClass() instancesRespondToSelector:@selector(eligibleForBTSmartRoutingConsideration)];
    if (!supportsEligibleForBT)
        return;

    RELEASE_LOG(Media, "AudioSession::setEligibleForSmartRouting() %s", eligible ? "true" : "false");

    AVAudioSession *session = [PAL::getAVAudioSessionClass() sharedInstance];
    if (session.eligibleForBTSmartRoutingConsideration == eligible)
        return;

    NSError *error = nil;
    if (![session setEligibleForBTSmartRoutingConsideration:eligible error:&error])
        RELEASE_LOG_ERROR(Media, "failed to set eligible to %d with error: %@", eligible, error.localizedDescription);
#else
    UNUSED_PARAM(eligible);
#endif
}

AudioSessionCocoa::AudioSessionCocoa()
    : m_workQueue(WorkQueue::create("AudioSession Activation Queue"_s))
{
}

AudioSessionCocoa::~AudioSessionCocoa()
{
    setEligibleForSmartRouting(false, ForceUpdate::Yes);
}

void AudioSessionCocoa::setEligibleForSmartRouting(bool isEligible, ForceUpdate forceUpdate)
{
    if (forceUpdate == ForceUpdate::No && m_isEligibleForSmartRouting == isEligible)
        return;

    m_isEligibleForSmartRouting = isEligible;
    m_workQueue->dispatch([this, isEligible] {
        setEligibleForSmartRoutingInternal(isEligible);
    });
}

bool AudioSessionCocoa::tryToSetActiveInternal(bool active)
{
#if HAVE(AVAUDIOSESSION)
    static bool supportsSharedInstance = [PAL::getAVAudioSessionClass() respondsToSelector:@selector(sharedInstance)];
    static bool supportsSetActive = [PAL::getAVAudioSessionClass() instancesRespondToSelector:@selector(setActive:withOptions:error:)];

    if (!supportsSharedInstance)
        return true;

    // We need to deactivate the session on another queue because the AVAudioSessionSetActiveOptionNotifyOthersOnDeactivation option
    // means that AVAudioSession may synchronously unduck previously ducked clients. Activation needs to complete before this method
    // returns, so do it synchronously on the same serial queue.
    if (active) {
        bool success = false;
        setEligibleForSmartRouting(true);
        m_workQueue->dispatchSync([&success] {
            NSError *error = nil;
            if (supportsSetActive)
                [[PAL::getAVAudioSessionClass() sharedInstance] setActive:YES withOptions:0 error:&error];
            if (error)
                RELEASE_LOG_ERROR(Media, "failed to activate audio session, error: %@", error.localizedDescription);
            success = !error;
        });
        return success;
    }

    m_workQueue->dispatch([] {
        NSError *error = nil;
        if (supportsSetActive)
            [[PAL::getAVAudioSessionClass() sharedInstance] setActive:NO withOptions:0 error:&error];
        if (error)
            RELEASE_LOG_ERROR(Media, "failed to deactivate audio session, error: %@", error.localizedDescription);
    });
    setEligibleForSmartRouting(false);
#else
    UNUSED_PARAM(active);
    notImplemented();
#endif
    return true;
}

void AudioSessionCocoa::setCategory(CategoryType newCategory, Mode, RouteSharingPolicy)
{
    // Disclaim support for Smart Routing when we are not generating audio.
    setEligibleForSmartRouting(isActive() && newCategory != AudioSessionCategory::None);
}

}

#endif // USE(AUDIO_SESSION) && PLATFORM(COCOA)
