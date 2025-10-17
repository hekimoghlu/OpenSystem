/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#import <WebKit/WKWebsiteDataRecord.h>

NS_ASSUME_NONNULL_BEGIN

@class _WKWebsiteDataSize;

WK_EXTERN NSString * const _WKWebsiteDataTypeHSTSCache WK_API_AVAILABLE(macos(10.11), ios(9.0));
WK_EXTERN NSString * const _WKWebsiteDataTypeMediaKeys WK_API_AVAILABLE(macos(10.11), ios(9.0));
// _WKWebsiteDataTypeSearchFieldRecentSearches will be deprecated; please use WKWebsiteDataTypeSearchFieldRecentSearches.
WK_EXTERN NSString * const _WKWebsiteDataTypeSearchFieldRecentSearches WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKWebsiteDataTypeResourceLoadStatistics WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKWebsiteDataTypeCredentials WK_API_AVAILABLE(macos(10.13), ios(11.0));
WK_EXTERN NSString * const _WKWebsiteDataTypeAdClickAttributions WK_API_AVAILABLE(macos(10.15), ios(13.0));
WK_EXTERN NSString * const _WKWebsiteDataTypePrivateClickMeasurements WK_API_AVAILABLE(macos(12.0), ios(15.0));
WK_EXTERN NSString * const _WKWebsiteDataTypeAlternativeServices WK_API_AVAILABLE(macos(11.0), ios(14.0));
WK_EXTERN NSString * const _WKWebsiteDataTypeFileSystem WK_API_DEPRECATED_WITH_REPLACEMENT("WKWebsiteDataTypeFileSystem", macos(13.0, 14.0), ios(16.0, 17.0));

@interface WKWebsiteDataRecord (WKPrivate)

@property (nullable, nonatomic, readonly) _WKWebsiteDataSize *_dataSize;

- (NSArray<NSString *> *)_originsStrings WK_API_AVAILABLE(macos(10.14), ios(12.0));

@end

NS_ASSUME_NONNULL_END
