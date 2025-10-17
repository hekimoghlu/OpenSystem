/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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

@import ObjectiveC;
@import CoreGraphics;

@protocol AVVideoCompositionInstruction<NSObject>
@required
@property (nonatomic, readonly) BOOL enablePostProcessing;
@end

@interface AVVideoCompositionInstruction : NSObject </*NSSecureCoding, NSCopying, NSMutableCopying,*/ AVVideoCompositionInstruction>

/* Indicates the background color of the composition. Solid BGRA colors only are supported; patterns and other color refs that are not supported will be ignored.
 If the background color is not specified the video compositor will use a default backgroundColor of opaque black.
 If the rendered pixel buffer does not have alpha, the alpha value of the backgroundColor will be ignored. */
@property (nonatomic, retain) __attribute__((NSObject)) CGColorRef backgroundColor;

@end
