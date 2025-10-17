/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
#import <WebKit/WKWindowFeatures.h>

NS_ASSUME_NONNULL_BEGIN

@interface WKWindowFeatures (WKPrivate)

@property (nonatomic, readonly) BOOL _wantsPopup WK_API_AVAILABLE(macos(14.2), ios(17.2));
@property (nonatomic, readonly) BOOL _hasAdditionalFeatures WK_API_AVAILABLE(macos(14.2), ios(17.2));
@property (nullable, nonatomic, readonly) NSNumber *_popup WK_API_AVAILABLE(macos(14.2), ios(17.2));
@property (nullable, nonatomic, readonly) NSNumber *_locationBarVisibility WK_API_AVAILABLE(macos(10.13), ios(11.0));
@property (nullable, nonatomic, readonly) NSNumber *_scrollbarsVisibility WK_API_AVAILABLE(macos(10.13), ios(11.0));
@property (nullable, nonatomic, readonly) NSNumber *_fullscreenDisplay WK_API_AVAILABLE(macos(10.13), ios(11.0));
@property (nullable, nonatomic, readonly) NSNumber *_dialogDisplay WK_API_AVAILABLE(macos(10.13), ios(11.0));

@end

NS_ASSUME_NONNULL_END
