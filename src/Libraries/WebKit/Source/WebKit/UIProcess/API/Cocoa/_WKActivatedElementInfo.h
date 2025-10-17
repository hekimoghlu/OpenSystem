/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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

#if TARGET_OS_IPHONE
@class UIImage;
#else
@class NSImage;
#endif

typedef NS_ENUM(NSInteger, _WKActivatedElementType) {
    _WKActivatedElementTypeLink,
    _WKActivatedElementTypeImage,
    _WKActivatedElementTypeAttachment WK_API_AVAILABLE(macos(10.12), ios(10.0)),
    _WKActivatedElementTypeUnspecified WK_API_AVAILABLE(macos(10.13), ios(11.0)),
} WK_API_AVAILABLE(macos(10.10), ios(8.0));

WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface _WKActivatedElementInfo : NSObject

@property (nonatomic, readonly) NSURL *URL;
@property (nonatomic, readonly) NSURL *imageURL;
@property (nonatomic, readonly) NSString *title;
@property (nonatomic, readonly) _WKActivatedElementType type;
@property (nonatomic, readonly) CGRect boundingRect;
@property (nonatomic, readonly) NSString *ID WK_API_AVAILABLE(macos(10.12), ios(10.0));
@property (nonatomic, readonly) BOOL isAnimatedImage WK_API_AVAILABLE(macos(10.15), ios(13.0));
#if defined(TARGET_OS_VISION) && TARGET_OS_VISION && __VISION_OS_VERSION_MIN_REQUIRED >= 20000
@property (nonatomic, readonly) BOOL isSpatialImage WK_API_AVAILABLE(visionos(2.2));
#endif // defined(TARGET_OS_VISION) && TARGET_OS_VISION & __VISION_OS_VERSION_MIN_REQUIRED >= 20000
#if TARGET_OS_IPHONE
@property (nonatomic, readonly) BOOL isAnimating;
@property (nonatomic, readonly) BOOL canShowAnimationControls;
@property (nonatomic, readonly) NSDictionary *userInfo WK_API_AVAILABLE(ios(11.0));
@property (nonatomic, readonly, copy) UIImage *image;
#else
@property (nonatomic, readonly, copy) NSImage *image;
#endif

@end
