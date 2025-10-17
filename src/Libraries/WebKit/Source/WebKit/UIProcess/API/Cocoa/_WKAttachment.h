/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, _WKAttachmentDisplayMode) {
    _WKAttachmentDisplayModeAuto = 1,
    _WKAttachmentDisplayModeInPlace,
    _WKAttachmentDisplayModeAsIcon
} WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

WK_CLASS_AVAILABLE(macos(10.13.4), ios(11.3))
@interface _WKAttachmentDisplayOptions : NSObject
@property (nonatomic) _WKAttachmentDisplayMode mode;
@end

WK_CLASS_AVAILABLE(macos(10.14), ios(12.0))
@interface _WKAttachmentInfo : NSObject
@property (nonatomic, readonly, nullable) NSString *contentType;
@property (nonatomic, readonly, nullable) NSString *name;
@property (nonatomic, readonly, nullable) NSString *filePath;
@property (nonatomic, readonly, nullable) NSData *data;
@property (nonatomic, readonly, nullable) NSFileWrapper *fileWrapper;
@property (nonatomic, readonly) BOOL shouldPreserveFidelity WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
@end

WK_CLASS_AVAILABLE(macos(10.13.4), ios(11.3))
@interface _WKAttachment : NSObject

- (void)setFileWrapper:(NSFileWrapper *)fileWrapper contentType:(nullable NSString *)contentType completion:(void(^ _Nullable)(NSError * _Nullable))completionHandler WK_API_AVAILABLE(macos(10.14.4), ios(12.2));

@property (nonatomic, readonly, nullable) _WKAttachmentInfo *info WK_API_AVAILABLE(macos(10.14.4), ios(12.2));
@property (nonatomic, readonly) NSString *uniqueIdentifier;
@property (nonatomic, readonly, getter=isConnected) BOOL connected WK_API_AVAILABLE(macos(10.14.4), ios(12.2));

// Deprecated SPI.
- (void)requestInfo:(void(^)(_WKAttachmentInfo * _Nullable, NSError * _Nullable))completionHandler WK_API_DEPRECATED_WITH_REPLACEMENT("-info", macos(10.14, 10.14.4), ios(12.0, 12.2));
- (void)setData:(NSData *)data newContentType:(nullable NSString *)newContentType newFilename:(nullable NSString *)newFilename completion:(void(^ _Nullable)(NSError * _Nullable))completionHandler WK_API_DEPRECATED_WITH_REPLACEMENT("Please use -setFileWrapper:contentType:completion: instead.", macos(10.13.4, 10.14.4), ios(11.3, 12.2));

@end

NS_ASSUME_NONNULL_END
