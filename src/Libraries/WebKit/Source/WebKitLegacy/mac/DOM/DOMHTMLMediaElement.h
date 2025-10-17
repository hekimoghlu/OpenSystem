/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 5, 2024.
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
#import <WebKitLegacy/DOMHTMLElement.h>

@class DOMMediaError;
@class DOMTimeRanges;
@class NSString;

enum {
    DOM_NETWORK_EMPTY = 0,
    DOM_NETWORK_IDLE = 1,
    DOM_NETWORK_LOADING = 2,
    DOM_NETWORK_NO_SOURCE = 3,
    DOM_HAVE_NOTHING = 0,
    DOM_HAVE_METADATA = 1,
    DOM_HAVE_CURRENT_DATA = 2,
    DOM_HAVE_FUTURE_DATA = 3,
    DOM_HAVE_ENOUGH_DATA = 4
} WEBKIT_ENUM_DEPRECATED_MAC(10_5, 10_14);

WEBKIT_CLASS_DEPRECATED_MAC(10_5, 10_14)
@interface DOMHTMLMediaElement : DOMHTMLElement
@property (readonly, strong) DOMMediaError *error;
@property (copy) NSString *src;
@property (readonly, copy) NSString *currentSrc;
@property (copy) NSString *crossOrigin;
@property (readonly) unsigned short networkState;
@property (copy) NSString *preload;
@property (readonly, strong) DOMTimeRanges *buffered;
@property (readonly) unsigned short readyState;
@property (readonly) BOOL seeking;
@property double currentTime;
@property (readonly) double duration;
@property (readonly) BOOL paused;
@property double defaultPlaybackRate;
@property double playbackRate;
@property (readonly, strong) DOMTimeRanges *played;
@property (readonly, strong) DOMTimeRanges *seekable;
@property (readonly) BOOL ended;
@property BOOL autoplay;
@property BOOL loop;
@property BOOL controls;
@property double volume;
@property BOOL muted;
@property BOOL defaultMuted;
@property BOOL webkitPreservesPitch;
@property (readonly) BOOL webkitHasClosedCaptions;
@property BOOL webkitClosedCaptionsVisible;
@property (copy) NSString *mediaGroup;

- (void)load;
- (NSString *)canPlayType:(NSString *)type;
- (NSTimeInterval)getStartDate;
- (void)play;
- (void)pause;
- (void)fastSeek:(double)time;
@end
