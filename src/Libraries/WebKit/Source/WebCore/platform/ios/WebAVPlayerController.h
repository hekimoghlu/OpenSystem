/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#if PLATFORM(COCOA) && HAVE(AVKIT)

#import <pal/spi/cocoa/AVKitSPI.h>

NS_ASSUME_NONNULL_BEGIN

namespace WebCore {
class PlaybackSessionModel;
class PlaybackSessionInterfaceIOS;
}

@class AVTimeRange;

@interface WebAVMediaSelectionOption : NSObject
- (instancetype)initWithMediaType:(AVMediaType)type displayName:(NSString *)displayName;

@property (nonatomic, readonly) NSString *displayName;
@property (nonatomic, readonly) NSString *localizedDisplayName;
@property (nonatomic, readonly) AVMediaType mediaType;

@end

@interface WebAVPlayerController : NSObject

- (void)setAllowsPictureInPicture:(BOOL)allowsPictureInPicture;

@property (retain) AVPlayerController *playerControllerProxy;
@property (assign, nullable /*weak*/) WebCore::PlaybackSessionModel* delegate;
@property (assign, nullable /*weak*/) WebCore::PlaybackSessionInterfaceIOS* playbackSessionInterface;

@property (readonly) BOOL canScanForward;
@property BOOL canScanBackward;
@property (readonly) BOOL canSeekToBeginning;
@property (readonly) BOOL canSeekToEnd;
@property (readonly) BOOL isScrubbing;
@property (readonly) BOOL canSeekFrameBackward;
@property (readonly) BOOL canSeekFrameForward;
@property (readonly) BOOL hasContentChapters;
@property (readonly) BOOL isSeeking;
@property (readonly) NSTimeInterval seekToTime;

@property BOOL canPlay;
@property (getter=isPlaying) BOOL playing;
@property BOOL canPause;
@property BOOL canTogglePlayback;
@property double defaultPlaybackRate;
@property double rate;
@property BOOL canSeek;
@property NSTimeInterval contentDuration;
@property CGSize contentDimensions;
@property BOOL hasEnabledAudio;
@property BOOL hasEnabledVideo;
@property BOOL hasVideo;
@property (readonly) NSTimeInterval minTime;
@property (readonly) NSTimeInterval maxTime;
@property NSTimeInterval contentDurationWithinEndTimes;
@property (retain) NSArray *loadedTimeRanges;
@property AVPlayerControllerStatus status;
@property (retain) AVValueTiming *timing;
@property (retain) NSArray *seekableTimeRanges;
@property (getter=isMuted) BOOL muted;
@property double volume;
- (void)volumeChanged:(double)volume;

@property (readonly) BOOL hasMediaSelectionOptions;
@property (readonly) BOOL hasAudioMediaSelectionOptions;
@property (retain) NSArray *audioMediaSelectionOptions;
@property (retain) WebAVMediaSelectionOption *currentAudioMediaSelectionOption;
@property (readonly) BOOL hasLegibleMediaSelectionOptions;
@property (retain) NSArray *legibleMediaSelectionOptions;
@property (retain) WebAVMediaSelectionOption *currentLegibleMediaSelectionOption;

@property (readonly, getter=isPlayingOnExternalScreen) BOOL playingOnExternalScreen;
@property (nonatomic, getter=isPlayingOnSecondScreen) BOOL playingOnSecondScreen;
@property (getter=isExternalPlaybackActive) BOOL externalPlaybackActive;
@property AVPlayerControllerExternalPlaybackType externalPlaybackType;
@property (retain) NSString *externalPlaybackAirPlayDeviceLocalizedName;
@property BOOL allowsExternalPlayback;
@property (readonly, getter=isPictureInPicturePossible) BOOL pictureInPicturePossible;
@property (getter=isPictureInPictureInterrupted) BOOL pictureInPictureInterrupted;

@property NSTimeInterval seekableTimeRangesLastModifiedTime;
@property NSTimeInterval liveUpdateInterval;

@property (NS_NONATOMIC_IOSONLY, retain, readwrite) AVValueTiming *minTiming;
@property (NS_NONATOMIC_IOSONLY, retain, readwrite) AVValueTiming *maxTiming;

- (void)setDefaultPlaybackRate:(double)defaultPlaybackRate fromJavaScript:(BOOL)fromJavaScript;
- (void)setRate:(double)rate fromJavaScript:(BOOL)fromJavaScript;

#if PLATFORM(APPLETV)
// FIXME (116592344): Remove these declarations once AVPlayerController API is available on tvOS.
@property (nonatomic, readonly, getter=isEffectiveRateNonZero) BOOL effectiveRateNonZero;
@property (nonatomic, readonly) CMTime forwardPlaybackEndTime;
@property (nonatomic, readonly) CMTime backwardPlaybackEndTime;
@property (nonatomic, readonly) BOOL isSeekingTV;
@property (nonatomic, readonly) BOOL hasStartAndEndDates;
@property (nonatomic, readonly, nullable) AVTimeRange *timeRangeSeekable;
@property (readonly, nullable) NSValue *overrideForForwardPlaybackEndTime;
@property (readonly, nullable) NSValue *overrideForReversePlaybackEndTime;
@property (readonly) double timebaseRate;
@property (readonly, nullable) NSArray *externalMetadata;
@property (readonly) BOOL isPlaybackLikelyToKeepUp;
@property (readonly) AVPlayerControllerTimeControlStatus timeControlStatus;
@property (readonly) NSTimeInterval displayedDuration;
@property (nonatomic, readonly) NSTimeInterval contentDurationCached;
@property (nonatomic, readonly) NSTimeInterval currentDisplayTime;
@property (nonatomic, readonly) NSDate *currentOrEstimatedDate;
@property (nonatomic, readonly) AVTimeRange *displayTimeRangeForNavigation;
@property (nonatomic, readonly) BOOL isContentDurationIndefinite;
@property (nonatomic, readonly) AVTimeRange *timeRangeForNavigation;
@property (nonatomic) float activeRate;
#endif // PLATFORM(APPLETV)

@end

Class webAVPlayerControllerClass();
RetainPtr<WebAVPlayerController> createWebAVPlayerController();

NS_ASSUME_NONNULL_END

#endif // PLATFORM(COCOA) && HAVE(AVKIT)
