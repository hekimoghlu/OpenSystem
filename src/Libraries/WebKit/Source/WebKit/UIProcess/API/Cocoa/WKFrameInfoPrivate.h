/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#import <WebKit/WKFrameInfo.h>

@class _WKFrameHandle;

@interface WKFrameInfo (WKPrivate)

@property (nonatomic, readonly, copy, nonnull) _WKFrameHandle *_handle WK_API_AVAILABLE(macos(10.12), ios(10.0));
@property (nonatomic, readonly, copy, nullable) _WKFrameHandle *_parentFrameHandle WK_API_AVAILABLE(macos(11.0), ios(14.0));
@property (nonatomic, readonly, copy, nullable) NSUUID *_documentIdentifier WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));
@property (nonatomic, readonly) pid_t _processIdentifier WK_API_AVAILABLE(macos(13.3), ios(16.4));
@property (nonatomic, readonly) BOOL _isLocalFrame WK_API_AVAILABLE(macos(14.0), ios(17.0));
@property (nonatomic, readonly) BOOL _isFocused WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));
@property (nonatomic, readonly) BOOL _errorOccurred WK_API_AVAILABLE(macos(14.4), ios(17.4), visionos(1.1));
@property (nonatomic, readonly, copy, nullable) NSString *_title WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
@property (nonatomic, readonly) BOOL _isScrollable WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));
@property (nonatomic, readonly) CGSize _contentSize WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));
@property (nonatomic, readonly) CGSize _visibleContentSize WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));
@property (nonatomic, readonly) CGSize _visibleContentSizeExcludingScrollbars WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));

@end
