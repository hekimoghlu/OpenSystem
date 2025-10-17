/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

@class WKWebViewConfiguration;
@class WKWebsiteDataStore;
@class WKWebExtensionController;

/*!
 @abstract A ``WKWebExtensionControllerConfiguration`` object with which to initialize a web extension controller.
 @discussion Contains properties used to configure a ``WKWebExtensionController``.
*/
WK_CLASS_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_SWIFT_UI_ACTOR NS_SWIFT_NAME(WKWebExtensionController.Configuration)
@interface WKWebExtensionControllerConfiguration : NSObject <NSSecureCoding, NSCopying>

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/*!
 @abstract Returns a new default configuration that is persistent and not unique.
 @discussion If a ``WKWebExtensionController`` is associated with a persistent configuration,
 data will be written to the file system in a common location. When using multiple extension controllers, each
 controller should use a unique configuration to avoid conflicts.
 @seealso configurationWithIdentifier:
*/
+ (instancetype)defaultConfiguration;

/*!
 @abstract Returns a new non-persistent configuration.
 @discussion If a ``WKWebExtensionController`` is associated with a non-persistent configuration,
 no data will be written to the file system. This is useful for extensions in "private browsing" situations.
*/
+ (instancetype)nonPersistentConfiguration;

/*!
 @abstract Returns a new configuration that is persistent and unique for the specified identifier.
 @discussion If a ``WKWebExtensionController`` is associated with a unique persistent configuration,
 data will be written to the file system in a unique location based on the specified identifier.
 @seealso defaultConfiguration
*/
+ (instancetype)configurationWithIdentifier:(NSUUID *)identifier;

/*! @abstract A Boolean value indicating if this context will write data to the the file system. */
@property (nonatomic, readonly, getter=isPersistent) BOOL persistent;

/*! @abstract The unique identifier used for persistent configuration storage, or `nil` when it is the default or not persistent. */
@property (nonatomic, nullable, readonly, copy) NSUUID *identifier;

/*! @abstract The web view configuration to be used as a basis for configuring web views in extension contexts. */
@property (nonatomic, null_resettable, copy) WKWebViewConfiguration *webViewConfiguration;

/*!
 @abstract The default data store for website data and cookie access in extension contexts.
 @discussion This property sets the primary data store for managing website data, including cookies, which extensions can access,
 subject to the granted permissions within the extension contexts. Defaults to ``WKWebsiteDataStore.defaultDataStore``.
 @note In addition to this data store, extensions can also access other data stores, such as non-persistent ones, for any open tabs.
 */
@property (nonatomic, null_resettable, retain) WKWebsiteDataStore *defaultWebsiteDataStore;

@end

WK_HEADER_AUDIT_END(nullability, sendability)
