/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#if PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)

#import <pal/spi/cocoa/AVKitSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>

namespace WebCore {
class PlaybackSessionInterfaceMac;
struct MediaSelectionOption;
}

#if ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)

WEBCORE_EXPORT
@interface WebPlaybackControlsManager : NSObject
    <AVTouchBarPlaybackControlsControlling>
{
@private
    NSTimeInterval _contentDuration;
    RetainPtr<AVValueTiming> _timing;
    NSTimeInterval _seekToTime;
    RetainPtr<NSArray> _seekableTimeRanges;
    RetainPtr<NSArray<AVTouchBarMediaSelectionOption *>> _audioTouchBarMediaSelectionOptions;
    RetainPtr<AVTouchBarMediaSelectionOption> _currentAudioTouchBarMediaSelectionOption;
    RetainPtr<NSArray<AVTouchBarMediaSelectionOption *>> _legibleTouchBarMediaSelectionOptions;
    RetainPtr<AVTouchBarMediaSelectionOption> _currentLegibleTouchBarMediaSelectionOption;
    RefPtr<WebCore::PlaybackSessionInterfaceMac> _playbackSessionInterfaceMac;
    double _defaultPlaybackRate;
    float _rate;
    BOOL _playing;
    BOOL _hasEnabledAudio;
    BOOL _hasEnabledVideo;
    BOOL _canTogglePlayback;
    BOOL _canSeek;
}

@property (assign) WebCore::PlaybackSessionInterfaceMac* playbackSessionInterfaceMac;
@property (readwrite) NSTimeInterval contentDuration;
@property (nonatomic, retain, readwrite) AVValueTiming *timing;
@property (nonatomic) NSTimeInterval seekToTime;
@property (nonatomic, retain, readwrite) NSArray *seekableTimeRanges;
@property (nonatomic) BOOL hasEnabledAudio;
@property (nonatomic) BOOL hasEnabledVideo;
@property (getter=isPlaying) BOOL playing;
@property BOOL canTogglePlayback;
@property double defaultPlaybackRate;
@property (nonatomic) float rate;
@property BOOL allowsPictureInPicturePlayback;
@property (getter=isPictureInPictureActive) BOOL pictureInPictureActive;
@property BOOL canTogglePictureInPicture;
- (void)togglePictureInPicture;
- (void)enterInWindow;
- (void)exitInWindow;
@property (nonatomic, readonly) BOOL canSeek;

- (AVTouchBarMediaSelectionOption *)currentAudioTouchBarMediaSelectionOption;
- (void)setCurrentAudioTouchBarMediaSelectionOption:(AVTouchBarMediaSelectionOption *)option;
- (AVTouchBarMediaSelectionOption *)currentLegibleTouchBarMediaSelectionOption;
- (void)setCurrentLegibleTouchBarMediaSelectionOption:(AVTouchBarMediaSelectionOption *)option;
- (void)setAudioMediaSelectionOptions:(const Vector<WebCore::MediaSelectionOption>&)options withSelectedIndex:(NSUInteger)selectedIndex;
- (void)setLegibleMediaSelectionOptions:(const Vector<WebCore::MediaSelectionOption>&)options withSelectedIndex:(NSUInteger)selectedIndex;
- (void)setAudioMediaSelectionIndex:(NSUInteger)selectedIndex;
- (void)setLegibleMediaSelectionIndex:(NSUInteger)selectedIndex;

- (void)setDefaultPlaybackRate:(double)defaultPlaybackRate fromJavaScript:(BOOL)fromJavaScript;
- (void)setRate:(double)rate fromJavaScript:(BOOL)fromJavaScript;
@end

#endif // ENABLE(WEB_PLAYBACK_CONTROLS_MANAGER)

#endif // PLATFORM(MAC) && ENABLE(VIDEO_PRESENTATION_MODE)
