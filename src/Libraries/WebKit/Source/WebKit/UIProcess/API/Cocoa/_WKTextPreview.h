/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
#import <WebKit/WKFoundation.h>

NS_SWIFT_SENDABLE
WK_CLASS_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2))
@interface _WKTextPreview : NSObject

// Preview image of text rendered against a transparent background.
@property (readonly) CGImageRef previewImage;

// The frame this image is meant to be presented in in the web view's coordinate space.
@property (readonly) CGRect presentationFrame;

- (instancetype)initWithSnapshotImage:(CGImageRef)snapshotImage presentationFrame:(CGRect)presentationFrame;

@end
