/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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

NS_ASSUME_NONNULL_BEGIN

@class WKWebView;
@class _WKInspectorExtension;

@protocol _WKInspectorExtensionHost <NSObject>
@optional

/**
 * @abstract Registers a Web Extension with the associated Web Inspector.
 * @param extensionID A unique identifier for the extension.
 * @param extensionBundleIdentifier A bundle identifier for the extension.
 * @param displayName A localized display name for the extension.
 * @param completionHandler The completion handler to be called when registration succeeds or fails.
 *
 * Web Extensions in Web Inspector are active as soon as they are registered.
 */
- (void)registerExtensionWithID:(NSString *)extensionID extensionBundleIdentifier:(NSString *)extensionBundleIdentifier displayName:(NSString *)displayName completionHandler:(void(^)(NSError * _Nullable, _WKInspectorExtension * _Nullable))completionHandler;

/**
 * @abstract Unregisters a Web Extension with the associated Web Inspector.
 * @param extensionID A unique identifier for the extension.
 * @param completionHandler The completion handler to be called when unregistering succeeds or fails.
 *
 * Unregistering an extension will automatically close any associated sidebars/tabs.
 */
- (void)unregisterExtension:(_WKInspectorExtension *)extension completionHandler:(void(^)(NSError * _Nullable))completionHandler;

/**
 * @abstract Opens the specified extension tab in the associated Web Inspector.
 * @param extensionTabIdentifier An identifier for an extension tab created using WKInspectorExtension methods.
 * @param completionHandler The completion handler to be called when the request to show the tab succeeds or fails.
 * @discussion This method has no effect if the extensionTabIdentifier is invalid.
 * It is an error to call this method prior to calling -[_WKInspectorIBActions show].
 */
- (void)showExtensionTabWithIdentifier:(NSString *)extensionTabIdentifier completionHandler:(void(^)(NSError * _Nullable))completionHandler;

/**
 * @abstract Loads the extension tab with a specified URL.
 * @param extensionTabIdentifier An identifier for an extension tab created using WKInspectorExtension methods.
 * @param url The URL that the should be loaded in the extension tab.
 * @param completionHandler The completion handler to be called when the load succeeds or fails.
 * @discussion This method has no effect if the extensionTabIdentifier is invalid.
 * It is an error to call this method prior to calling -[_WKInspectorIBActions show].
 */
- (void)navigateExtensionTabWithIdentifier:(NSString *)extensionTabIdentifier toURL:(NSURL *)url completionHandler:(void(^)(NSError * _Nullable))completionHandler;

/**
 * @abstract The web view that is used to host extension tabs created via _WKInspectorExtension.
 * @discussion Browsing contexts for extension tabs are loaded in subframes of this web view.
 */
@property (nonatomic, nullable, readonly) WKWebView *extensionHostWebView;
@end

NS_ASSUME_NONNULL_END
