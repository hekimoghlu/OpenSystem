/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#if TARGET_OS_OSX || TARGET_OS_IOS || (defined(TARGET_OS_VISION) && TARGET_OS_VISION)

#import <Foundation/Foundation.h>

@class WKFrameInfo;

typedef NS_ENUM(NSInteger, _WKHitTestResultElementType) {
    _WKHitTestResultElementTypeNone,
    _WKHitTestResultElementTypeAudio,
    _WKHitTestResultElementTypeVideo,
} WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

WK_CLASS_AVAILABLE(macos(10.12), ios(16.0))
@interface _WKHitTestResult : NSObject <NSCopying>

@property (nonatomic, readonly, copy) NSURL *absoluteImageURL;
@property (nonatomic, readonly, copy) NSString *imageMIMEType WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

@property (nonatomic, readonly, copy) NSURL *absolutePDFURL;
@property (nonatomic, readonly, copy) NSURL *absoluteLinkURL;
@property (nonatomic, readonly) BOOL hasLocalDataForLinkURL WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
@property (nonatomic, readonly, copy) NSString *linkLocalDataMIMEType WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
@property (nonatomic, readonly, copy) NSURL *absoluteMediaURL;

@property (nonatomic, readonly, copy) NSString *linkLabel;
@property (nonatomic, readonly, copy) NSString *linkTitle;
@property (nonatomic, readonly, copy) NSString *linkSuggestedFilename WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
@property (nonatomic, readonly, copy) NSString *imageSuggestedFilename WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
@property (nonatomic, readonly, copy) NSString *lookupText;

@property (nonatomic, readonly, getter=isContentEditable) BOOL contentEditable;
@property (nonatomic, readonly, getter=isSelected) BOOL selected WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));
@property (nonatomic, readonly, getter=isMediaDownloadable) BOOL mediaDownloadable WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));
@property (nonatomic, readonly, getter=isMediaFullscreen) BOOL mediaFullscreen WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

@property (nonatomic, readonly) CGRect elementBoundingBox;

@property (nonatomic, readonly) _WKHitTestResultElementType elementType WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

@property (nonatomic, readonly) WKFrameInfo *frameInfo WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));

@end

#endif
