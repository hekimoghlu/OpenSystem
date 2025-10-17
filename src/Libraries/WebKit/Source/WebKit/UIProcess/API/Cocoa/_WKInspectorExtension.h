/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#pragma once

#import <WebKit/WKFoundation.h>

#if TARGET_OS_OSX

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@protocol _WKInspectorExtensionDelegate;

WK_CLASS_AVAILABLE(macos(12.0))
@interface _WKInspectorExtension : NSObject

- (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/**
 * @abstract Creates a new tab in the Web Inspector interface for this extension.
 * @param tabName A localized display name for the tab.
 * @param tabIconURL The location of an image resource to use for display in the created tab's title.
 * @param sourceURL The location of the main resource to load in the new tab's iframe browsing context.
 * @param completionHandler The completion handler to be called when creating a tab succeeds or fails.
 * @discussion The inspectorTabIdentifier provided to the completion handler can be used to identify tabs
 * that are passed as a parameter to _WKInspectorExtensionDelegate methods.
 */
- (void)createTabWithName:(NSString *)tabName tabIconURL:(NSURL *)tabIconURL sourceURL:(NSURL *)sourceURL completionHandler:(void(^)(NSError * _Nullable, NSString * _Nullable inspectorTabIdentifier))completionHandler;

/**
 * @abstract Evaluates JavaScript in the context of the inspected page on behalf of the _WKInspectorExtension.
 * @param scriptSource The JavaScript code to be evaluated.
 * @param frameURL URL for the frame in which to evaluate the script. If nil is passed, the main frame will be used.
 * @param contextSecurityOrigin If specified, evaluate the script in the content script context of a different security origin.
 * @param usesContentScriptContext If YES, evaluate the script in the context of the extension's content scripts.
 * @param completionHandler A block to invoke when the operation completes or fails.
 * @discussion The completionHandler is passed an NSJSONSerialization-compatible NSObject representing the evaluation result, or an error.
 * scriptSource is treated as a top-level evaluation. By default, the script is evaluated in the inspected page's script context.
 * The inspected page ultimately controls its execution context and the result of this evaluation. Thus, the result shall be treated as untrusted input.
 */
- (void)evaluateScript:(NSString *)scriptSource frameURL:(NSURL * _Nullable)frameURL contextSecurityOrigin:(NSURL * _Nullable)contextSecurityOrigin useContentScriptContext:(BOOL)useContentScriptContext completionHandler:(void(^)(NSError * _Nullable, id result))completionHandler;

/**
 * @abstract Evaluates JavaScript in the context of a Web Inspector tab created by this _WKInspectorExtension.
 * @param scriptSource The JavaScript code to be evaluated.
 * @param tabIdentifier Identifier for the Web Inspector tab in which to evaluate JavaScript.
 * @param completionHandler A block to invoke when the operation completes or fails.
 * @discussion The completionHandler is passed an NSJSONSerialization-compatible NSObject representing the evaluation result, or an error.
 * scriptSource is treated as a top-level evaluation.
 */
- (void)evaluateScript:(NSString *)scriptSource inTabWithIdentifier:(NSString *)tabIdentifier completionHandler:(void(^)(NSError * _Nullable, id result))completionHandler;

/**
 * @abstract Navigates a tab created by this _WKInspectorExtension to a new URL.
 * @param url The url to be loaded.
 * @param tabIdentifier Identifier for the Web Inspector tab in which to navigate.
 * @param completionHandler A block to invoke when the operation completes or fails.
 */
- (void)navigateToURL:(NSURL *)url inTabWithIdentifier:(NSString *)tabIdentifier completionHandler:(void(^)(NSError * _Nullable))completionHandler;

/**
 * @abstract Reloads the inspected page on behalf of the _WKInspectorExtension.
 * @param ignoreCache If YES, reloads the page while ignoring the cache.
 * @param userAgent If specified, overrides the user agent to be sent in the `User-Agent` header and returned by calls to `navigator.userAgent` made by scripts running in the page. This only affects the next navigation.
 * @param injectedScript If specified, injects the given JavaScript expression into all frames on the page before any other scripts.
 * @param completionHandler A block to invoke when the operation completes or fails.
 */
- (void)reloadIgnoringCache:(BOOL)ignoreCache userAgent:(NSString *)userAgent injectedScript:(NSString *)injectedScript completionHandler:(void(^)(NSError * _Nullable))completionHandler;

@property (readonly, nonatomic) NSString *extensionID;

/**
 * @abstract Allows the client to receive extension lifecycle events that
 * arise from within Web Inspector.
 */
@property (nonatomic, weak) id <_WKInspectorExtensionDelegate> delegate WK_API_AVAILABLE(macos(12.0));

@end

NS_ASSUME_NONNULL_END

#endif // TARGET_OS_OSX
