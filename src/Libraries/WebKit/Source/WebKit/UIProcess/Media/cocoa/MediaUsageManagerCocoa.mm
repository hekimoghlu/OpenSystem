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
#import "config.h"
#import "MediaUsageManagerCocoa.h"

#if ENABLE(MEDIA_USAGE)

#import <WebCore/NotImplemented.h>
#import <pal/cocoa/UsageTrackingSoftLink.h>

NS_ASSUME_NONNULL_BEGIN

@interface USVideoUsage : NSObject
- (instancetype)initWithBundleIdentifier:(NSString *)bundleIdentifier URL:(NSURL *)url mediaURL:(NSURL *)mediaURL videoMetadata:(NSDictionary<NSString *, id> *)videoMetadata NS_DESIGNATED_INITIALIZER;
- (void)stop;
- (void)restart;
- (void)updateVideoMetadata:(NSDictionary<NSString *, id> *)videoMetadata;
@end

NS_ASSUME_NONNULL_END

namespace WebKit {
using namespace WebCore;

static bool usageTrackingAvailable()
{
    static bool available;

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [&] () {
        available = PAL::isUsageTrackingFrameworkAvailable()
            && PAL::getUSVideoUsageClass()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyCanShowControlsManager()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyCanShowNowPlayingControls()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsSuspended()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsInActiveDocument()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsFullscreen()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsMuted()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsMediaDocumentInMainFrame()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsAudio()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyAudioElementWithUserGesture()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyUserHasPlayedAudioBefore()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsElementRectMostlyInMainFrame()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyNoAudio()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyPlaybackPermitted()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyPageMediaPlaybackSuspended()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsMediaDocumentAndNotOwnerElement()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyPageExplicitlyAllowsElementToAutoplayInline()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyRequiresFullscreenForVideoPlaybackAndFullscreenNotPermitted()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoRateChange()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsAudioAndRequiresUserGestureForAudioRateChange()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoDueToLowPowerMode()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyNoUserGestureRequired()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyRequiresPlaybackAndIsNotPlaying()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyHasEverNotifiedAboutPlaying()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyOutsideOfFullscreen()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsVideo()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyRenderer()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyNoVideo()
            && PAL::canLoad_UsageTracking_USVideoMetadataKeyIsLargeEnoughForMainContent();
    });

    return available;
}

std::unique_ptr<MediaUsageManager> MediaUsageManager::create()
{
    return makeUnique<MediaUsageManagerCocoa>();
}

#if PLATFORM(COCOA) && !HAVE(CGS_FIX_FOR_RADAR_97530095)
bool MediaUsageManager::isPlayingVideoInViewport() const
{
    notImplemented();
    return false;
}
#endif

MediaUsageManagerCocoa::~MediaUsageManagerCocoa()
{
    reset();
}

void MediaUsageManagerCocoa::reset()
{
    for (auto& session : m_mediaSessions.values()) {
        if (session->usageTracker && session->mediaUsageInfo && session->mediaUsageInfo->isPlaying)
            [session->usageTracker stop];
    }
    m_mediaSessions.clear();
}

void MediaUsageManagerCocoa::addMediaSession(WebCore::MediaSessionIdentifier identifier, const String& bundleIdentifier, const URL& pageURL)
{
    auto addResult = m_mediaSessions.ensure(identifier, [&] {
        return makeUnique<MediaUsageManagerCocoa::SessionMediaUsage>(identifier, bundleIdentifier, pageURL);
    });
    ASSERT_UNUSED(addResult, addResult.isNewEntry);
}

void MediaUsageManagerCocoa::removeMediaSession(WebCore::MediaSessionIdentifier identifier)
{
    ASSERT(m_mediaSessions.contains(identifier));
    m_mediaSessions.remove(identifier);
}

void MediaUsageManagerCocoa::updateMediaUsage(WebCore::MediaSessionIdentifier identifier, const WebCore::MediaUsageInfo& mediaUsageInfo)
{
    ASSERT(m_mediaSessions.contains(identifier));
    auto session = m_mediaSessions.get(identifier);
    if (!session)
        return;

    if (!usageTrackingAvailable())
        return;

    @try {

        if (session->mediaUsageInfo) {
            if (session->mediaUsageInfo == mediaUsageInfo)
                return;

            if (session->usageTracker && session->mediaUsageInfo->mediaURL != mediaUsageInfo.mediaURL) {
                [session->usageTracker stop];
                session->usageTracker = nullptr;
            }
        }

        NSDictionary<NSString *, id> *metadata = @{
            USVideoMetadataKeyCanShowControlsManager: @(mediaUsageInfo.canShowControlsManager),
            USVideoMetadataKeyCanShowNowPlayingControls: @(mediaUsageInfo.canShowNowPlayingControls),
            USVideoMetadataKeyIsSuspended: @(mediaUsageInfo.isSuspended),
            USVideoMetadataKeyIsInActiveDocument: @(mediaUsageInfo.isInActiveDocument),
            USVideoMetadataKeyIsFullscreen: @(mediaUsageInfo.isFullscreen),
            USVideoMetadataKeyIsMuted: @(mediaUsageInfo.isMuted),
            USVideoMetadataKeyIsMediaDocumentInMainFrame: @(mediaUsageInfo.isMediaDocumentInMainFrame),
            USVideoMetadataKeyIsVideo: @(mediaUsageInfo.isVideo),
            USVideoMetadataKeyIsAudio: @(mediaUsageInfo.isAudio),
            USVideoMetadataKeyNoVideo: @(!mediaUsageInfo.hasVideo),
            USVideoMetadataKeyNoAudio: @(!mediaUsageInfo.hasAudio),
            USVideoMetadataKeyRenderer: @(mediaUsageInfo.hasRenderer),
            USVideoMetadataKeyAudioElementWithUserGesture: @(mediaUsageInfo.audioElementWithUserGesture),
            USVideoMetadataKeyUserHasPlayedAudioBefore: @(mediaUsageInfo.userHasPlayedAudioBefore),
            USVideoMetadataKeyIsElementRectMostlyInMainFrame: @(mediaUsageInfo.isElementRectMostlyInMainFrame),
            USVideoMetadataKeyPlaybackPermitted: @(mediaUsageInfo.playbackPermitted),
            USVideoMetadataKeyPageMediaPlaybackSuspended: @(mediaUsageInfo.pageMediaPlaybackSuspended),
            USVideoMetadataKeyIsMediaDocumentAndNotOwnerElement: @(mediaUsageInfo.isMediaDocumentAndNotOwnerElement),
            USVideoMetadataKeyPageExplicitlyAllowsElementToAutoplayInline: @(mediaUsageInfo.pageExplicitlyAllowsElementToAutoplayInline),
            USVideoMetadataKeyRequiresFullscreenForVideoPlaybackAndFullscreenNotPermitted: @(mediaUsageInfo.requiresFullscreenForVideoPlaybackAndFullscreenNotPermitted),
            USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoRateChange: @(mediaUsageInfo.isVideoAndRequiresUserGestureForVideoRateChange),
            USVideoMetadataKeyIsAudioAndRequiresUserGestureForAudioRateChange: @(mediaUsageInfo.isAudioAndRequiresUserGestureForAudioRateChange),
            USVideoMetadataKeyIsVideoAndRequiresUserGestureForVideoDueToLowPowerMode: @(mediaUsageInfo.isVideoAndRequiresUserGestureForVideoDueToLowPowerMode),
            USVideoMetadataKeyNoUserGestureRequired: @(mediaUsageInfo.noUserGestureRequired),
            USVideoMetadataKeyRequiresPlaybackAndIsNotPlaying: @(mediaUsageInfo.requiresPlaybackAndIsNotPlaying),
            USVideoMetadataKeyHasEverNotifiedAboutPlaying: @(mediaUsageInfo.hasEverNotifiedAboutPlaying),
            USVideoMetadataKeyOutsideOfFullscreen: @(mediaUsageInfo.outsideOfFullscreen),
            USVideoMetadataKeyIsLargeEnoughForMainContent: @(mediaUsageInfo.isLargeEnoughForMainContent),
        };

        if (!session->usageTracker) {
            if (!mediaUsageInfo.isPlaying)
                return;

            session->usageTracker = adoptNS([PAL::allocUSVideoUsageInstance() initWithBundleIdentifier:session->bundleIdentifier URL:(NSURL *)session->pageURL
                mediaURL:(NSURL *)mediaUsageInfo.mediaURL videoMetadata:metadata]);
            ASSERT(session->usageTracker);
            if (!session->usageTracker)
                return;
        } else
            [session->usageTracker updateVideoMetadata:metadata];

        if (session->mediaUsageInfo && session->mediaUsageInfo->isPlaying != mediaUsageInfo.isPlaying) {
            if (mediaUsageInfo.isPlaying)
                [session->usageTracker restart];
            else
                [session->usageTracker stop];
        }

        session->mediaUsageInfo = mediaUsageInfo;

    } @catch(NSException *exception) {
        WTFLogAlways("MediaUsageManagerCocoa::updateMediaUsage caught exception: %@", [[exception reason] UTF8String]);
    }
}

#if !HAVE(CGS_FIX_FOR_RADAR_97530095)
bool MediaUsageManagerCocoa::isPlayingVideoInViewport() const
{
    for (auto& session : m_mediaSessions.values()) {
        if (session->mediaUsageInfo && session->mediaUsageInfo->isPlaying && session->mediaUsageInfo->isVideo && session->mediaUsageInfo->isInViewport)
            return true;
    }
    return false;
}
#endif

} // namespace WebKit

#endif // ENABLE(MEDIA_USAGE)

