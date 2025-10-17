/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#pragma once

#if PLATFORM(COCOA)

#import <Foundation/Foundation.h>

OBJC_CLASS NSImage;
OBJC_CLASS UIImage;

namespace WebCore {

#if USE(APPKIT)
using CocoaImage = NSImage;
#else
using CocoaImage = UIImage;
#endif

// Returns an NSImage or UIImage depending on platform
CocoaImage *webCoreTextAttachmentMissingPlatformImage();
bool isWebCoreTextAttachmentMissingPlatformImage(CocoaImage *);

} // namespace WebCore

#if !PLATFORM(IOS_FAMILY)
@interface NSTextAttachment (WebCoreNSTextAttachment)
- (void)setIgnoresOrientation:(BOOL)flag;
- (void)setBounds:(CGRect)bounds;
- (BOOL)ignoresOrientation;
@property (strong) NSString *accessibilityLabel;
@end
#endif

#endif // PLATFORM(COCOA)
