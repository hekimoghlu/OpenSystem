/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#if TARGET_OS_IPHONE

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <WebKitLegacy/WebFrameIOS.h>

@interface WebSelectionRect : NSObject <NSCopying>

@property (nonatomic, assign) CGRect rect;
@property (nonatomic, assign) WKWritingDirection writingDirection;
@property (nonatomic, assign) BOOL isLineBreak;
@property (nonatomic, assign) BOOL isFirstOnLine;
@property (nonatomic, assign) BOOL isLastOnLine;
@property (nonatomic, assign) BOOL containsStart;
@property (nonatomic, assign) BOOL containsEnd;
@property (nonatomic, assign) BOOL isInFixedPosition;
@property (nonatomic, assign) BOOL isHorizontal;

+ (WebSelectionRect *)selectionRect;

+ (CGRect)startEdge:(NSArray *)rects;
+ (CGRect)endEdge:(NSArray *)rects;

@end

#endif // TARGET_OS_IPHONE
