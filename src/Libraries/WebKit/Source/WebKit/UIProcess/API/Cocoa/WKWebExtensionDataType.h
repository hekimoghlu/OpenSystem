/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

/*! @abstract Constants for specifying data types for a ``WKWebExtensionDataRecord``. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
typedef NSString * WKWebExtensionDataType NS_TYPED_ENUM NS_SWIFT_NAME(WKWebExtension.DataType);

/*! @abstract Specifies local storage, including `browser.storage.local`. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionDataType const WKWebExtensionDataTypeLocal NS_SWIFT_NONISOLATED;

/*! @abstract Specifies session storage, including `browser.storage.session`. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionDataType const WKWebExtensionDataTypeSession NS_SWIFT_NONISOLATED;

/*! @abstract Specifies synchronized storage, including `browser.storage.sync`. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN WKWebExtensionDataType const WKWebExtensionDataTypeSynchronized NS_SWIFT_NONISOLATED;
