/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#import <WebKit/WKWebExtensionDataType.h>

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

/*! @abstract Indicates a ``WKWebExtensionDataRecord`` error. */
WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_EXTERN NSErrorDomain const WKWebExtensionDataRecordErrorDomain NS_SWIFT_NAME(WKWebExtensionDataRecord.errorDomain) NS_SWIFT_NONISOLATED;

/*!
 @abstract Constants used by ``NSError`` to indicate errors in the ``WKWebExtensionDataRecord`` domain.
 @constant WKWebExtensionDataRecordErrorUnknown  Indicates that an unknown error occurred.
 @constant WKWebExtensionDataRecordErrorLocalStorageFailed  Indicates a failure occurred when either deleting or calculating local storage.
 @constant WKWebExtensionDataRecordErrorSessionStorageFailed  Indicates a failure occurred when either deleting or calculating session storage.
 @constant WKWebExtensionDataRecordErrorSynchronizedStorageFailed  Indicates a failure occurred when either deleting or calculating synchronized storage.
 */
typedef NS_ERROR_ENUM(WKWebExtensionDataRecordErrorDomain, WKWebExtensionDataRecordError) {
    WKWebExtensionDataRecordErrorUnknown = 1,
    WKWebExtensionDataRecordErrorLocalStorageFailed,
    WKWebExtensionDataRecordErrorSessionStorageFailed,
    WKWebExtensionDataRecordErrorSynchronizedStorageFailed,
} NS_SWIFT_NAME(WKWebExtensionDataRecord.Error) WK_API_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA));

/*!
 @abstract A ``WKWebExtensionDataRecord`` object represents a record of stored data for a specific web extension context.
 @discussion Contains properties and methods to query the data types and sizes.
*/
WK_CLASS_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_SWIFT_UI_ACTOR NS_SWIFT_NAME(WKWebExtension.DataRecord)
@interface WKWebExtensionDataRecord : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/*! @abstract The display name for the web extension to which this data record belongs. */
@property (nonatomic, readonly, copy) NSString *displayName;

/*! @abstract Unique identifier for the web extension context to which this data record belongs. */
@property (nonatomic, readonly, copy) NSString *uniqueIdentifier;

/*! @abstract The set of data types contained in this data record. */
@property (nonatomic, readonly, copy) NSSet<WKWebExtensionDataType> *containedDataTypes;

/*! @abstract An array of errors that may have occurred when either calculating or deleting storage. */
@property (nonatomic, readonly, copy) NSArray<NSError *> *errors;

/*!
 @abstract The total size in bytes of all data types contained in this data record.
 @seealso sizeInBytesOfTypes:
 */
@property (nonatomic, readonly) NSUInteger totalSizeInBytes;

/*!
 @abstract Retrieves the size in bytes of the specific data types in this data record.
 @param dataTypes The set of data types to measure the size for.
 @return The total size of the specified data types.
 @seealso totalSizeInBytes
 */
- (NSUInteger)sizeInBytesOfTypes:(NSSet<WKWebExtensionDataType> *)dataTypes NS_SWIFT_NAME(sizeInBytes(ofTypes:));

@end

WK_HEADER_AUDIT_END(nullability, sendability)
