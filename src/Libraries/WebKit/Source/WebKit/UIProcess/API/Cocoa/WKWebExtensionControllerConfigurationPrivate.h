/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#import <WebKit/WKWebExtensionControllerConfiguration.h>

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

@interface WKWebExtensionControllerConfiguration ()

/*!
 @abstract Returns a new configuration that is persistent and uses a temporary directory.
 @discussion This method creates a configuration for a ``WKWebExtensionController`` that is persistent during the session
 and uses a temporary directory for storage. This is ideal for scenarios that require temporary data persistence, such as testing.
 Each instance is created with a unique temporary directory.
*/
+ (instancetype)_temporaryConfiguration;

/*!
 @abstract A Boolean value indicating if this configuration uses a temporary directory.
 @discussion Indicates whether the configuration is persistent, with data stored in a temporary directory.
*/
@property (nonatomic, readonly, getter=_isTemporary) BOOL _temporary;

/*!
 @abstract The file path to the storage directory, if applicable.
 @discussion Provides the file path to the storage directory used by the configuration. It is `nil` for non-persistent
 configurations. For persistent configurations, it provides the path where data is stored, which may be a temporary directory.
*/
@property (nonatomic, nullable, copy, setter=_setStorageDirectoryPath:) NSString *_storageDirectoryPath;

@end

WK_HEADER_AUDIT_END(nullability, sendability)
