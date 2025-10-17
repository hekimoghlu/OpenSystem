/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_IPHONE

/*! @enum WKDataDetectorTypes
 @abstract The type of data detected.
 @constant WKDataDetectorTypeNone No detection is performed.
 @constant WKDataDetectorTypePhoneNumber Phone numbers are detected and turned into links.
 @constant WKDataDetectorTypeLink URLs in text are detected and turned into links.
 @constant WKDataDetectorTypeAddress Addresses are detected and turned into links.
 @constant WKDataDetectorTypeCalendarEvent Dates and times that are in the future are detected and turned into links.
 @constant WKDataDetectorTypeAll All of the above data types are turned into links when detected. Choosing this value will
 automatically include any new detection type that is added.
 */
typedef NS_OPTIONS(NSUInteger, WKDataDetectorTypes) {
    WKDataDetectorTypeNone = 0,
    WKDataDetectorTypePhoneNumber = 1 << 0,
    WKDataDetectorTypeLink = 1 << 1,
    WKDataDetectorTypeAddress = 1 << 2,
    WKDataDetectorTypeCalendarEvent = 1 << 3,
    WKDataDetectorTypeTrackingNumber = 1 << 4,
    WKDataDetectorTypeFlightNumber = 1 << 5,
    WKDataDetectorTypeLookupSuggestion = 1 << 6,
    WKDataDetectorTypeAll = NSUIntegerMax,

    WKDataDetectorTypeSpotlightSuggestion WK_API_DEPRECATED_WITH_REPLACEMENT("WKDataDetectorTypeLookupSuggestion", ios(10.0, 10.0)) = WKDataDetectorTypeLookupSuggestion,
} WK_API_AVAILABLE(ios(10.0));

#endif

NS_ASSUME_NONNULL_END
