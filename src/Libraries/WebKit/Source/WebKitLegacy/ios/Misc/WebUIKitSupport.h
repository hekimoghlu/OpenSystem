/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#import <CoreGraphics/CoreGraphics.h>
#import <unicode/uchar.h>

#ifdef __cplusplus
extern "C" {
#endif

// To be able to use background tasks from within WebKit, we need to expose that UIKit functionality
// without linking to UIKit.
// We accomplish this by giving UIKit 3 methods to set up:
//   - The invalid task ID value
//   - A block for starting a background task
//   - A block for ending a background task.
typedef NSUInteger WebBackgroundTaskIdentifier;
typedef void (^VoidBlock)(void);
typedef WebBackgroundTaskIdentifier (^StartBackgroundTaskBlock)(VoidBlock);
typedef void (^EndBackgroundTaskBlock)(WebBackgroundTaskIdentifier);

void WebKitSetInvalidWebBackgroundTaskIdentifier(WebBackgroundTaskIdentifier);
void WebKitSetStartBackgroundTaskBlock(StartBackgroundTaskBlock);
void WebKitSetEndBackgroundTaskBlock(EndBackgroundTaskBlock);

// This method gives WebKit the notifications to listen to so it knows about app Suspend/Resume
void WebKitSetBackgroundAndForegroundNotificationNames(NSString *, NSString *);

void WebKitInitialize(void);
float WebKitGetMinimumZoomFontSize(void);
    
int WebKitGetLastLineBreakInBuffer(UChar *characters, int position, int length);

const char *WebKitPlatformSystemRootDirectory(void);

CGPathRef WebKitCreatePathWithShrinkWrappedRects(NSArray* cgRects, CGFloat radius);

#ifdef __cplusplus
}
#endif
