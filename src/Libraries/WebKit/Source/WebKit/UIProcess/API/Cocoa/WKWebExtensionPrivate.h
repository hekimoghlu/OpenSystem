/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
#import <WebKit/WKWebExtension.h>

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

@interface WKWebExtension ()

- (nullable instancetype)_initWithAppExtensionBundle:(NSBundle *)appExtensionBundle error:(NSError **)error NS_SWIFT_UNAVAILABLE("Use init(appExtensionBundle:).");
- (nullable instancetype)_initWithResourceBaseURL:(NSURL *)resourceBaseURL error:(NSError **)error NS_SWIFT_UNAVAILABLE("Use init(resourceBaseURL:).");
- (nullable instancetype)_initWithAppExtensionBundle:(NSBundle *)appExtensionBundle resourceBaseURL:(NSURL *)resourceBaseURL error:(NSError **)error NS_SWIFT_UNAVAILABLE("Use init(appExtensionBundle:resourceBaseURL:).");
- (nullable instancetype)_initWithManifestDictionary:(NSDictionary<NSString *, id> *)manifest NS_SWIFT_UNAVAILABLE("Use init(manifestDictionary:).");
- (nullable instancetype)_initWithManifestDictionary:(NSDictionary<NSString *, id> *)manifest resources:(nullable NSDictionary<NSString *, id> *)resources NS_SWIFT_UNAVAILABLE("Use init(manifestDictionary:resources:).");
- (nullable instancetype)_initWithResources:(NSDictionary<NSString *, id> *)resources NS_SWIFT_UNAVAILABLE("Use init(resources:).");

/*!
 @abstract Returns a web extension initialized with a specified app extension bundle.
 @param appExtensionBundle The bundle to use for the new web extension.
 @param error Set to \c nil or an error instance if an error occurred.
 @result An initialized web extension, or `nil` if the object could not be initialized due to an error.
 */
- (nullable instancetype)initWithAppExtensionBundle:(NSBundle *)appExtensionBundle error:(NSError **)error NS_REFINED_FOR_SWIFT;

/*!
 @abstract Returns a web extension initialized with a specified resource base URL.
 @param resourceBaseURL The directory URL to use for the new web extension.
 @param error Set to \c nil or an error instance if an error occurred.
 @result An initialized web extension, or `nil` if the object could not be initialized due to an error.
 @discussion The URL must be a file URL that points to either a directory containing a `manifest.json` file or a valid ZIP archive.
 */
- (nullable instancetype)initWithResourceBaseURL:(NSURL *)resourceBaseURL error:(NSError **)error NS_REFINED_FOR_SWIFT;

/*!
 @abstract Returns a web extension initialized with a specified app extension bundle and resource base URL.
 @param appExtensionBundle The bundle to use for the new web extension. Can be \c nil if a resource base URL is provided.
 @param resourceBaseURL The directory URL to use for the new web extension. Can be \c nil if an app extension bundle is provided.
 @param error Set to \c nil or an error instance if an error occurred.
 @result An initialized web extension, or `nil` if the object could not be initialized due to an error.
 @discussion Either the app extension bundle or the resource base URL (which can point to a directory or a valid ZIP archive) must be provided.
 This initializer is useful when the extension resources are in a different location from the app extension bundle used for native messaging.
 */
- (nullable instancetype)initWithAppExtensionBundle:(nullable NSBundle *)appExtensionBundle resourceBaseURL:(nullable NSURL *)resourceBaseURL error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/*!
 @abstract Returns a web extension initialized with a specified manifest dictionary.
 @param manifest The dictionary containing the manifest data for the web extension.
 @result An initialized web extension, or `nil` if the object could not be initialized due to an error.
 */
- (nullable instancetype)initWithManifestDictionary:(NSDictionary<NSString *, id> *)manifest;

/*!
 @abstract Returns a web extension initialized with a specified manifest dictionary and resources.
 @param manifest The dictionary containing the manifest data for the web extension.
 @param resources A dictionary of file paths to data, string, or JSON-serializable values.
 @result An initialized web extension, or `nil` if the object could not be initialized due to an error.
 @discussion The resources dictionary provides additional data required for the web extension. Paths in resources can
 have subdirectories, such as `_locales/en/messages.json`.
 */
- (nullable instancetype)initWithManifestDictionary:(NSDictionary<NSString *, id> *)manifest resources:(nullable NSDictionary<NSString *, id> *)resources NS_DESIGNATED_INITIALIZER;

/*!
 @abstract Returns a web extension initialized with specified resources.
 @param resources A dictionary of file paths to data, string, or JSON-serializable values.
 @result An initialized web extension, or `nil` if the object could not be initialized due to an error.
 @discussion The resources dictionary must provide at least the `manifest.json` resource.  Paths in resources can
 have subdirectories, such as `_locales/en/messages.json`.
 */
- (nullable instancetype)initWithResources:(NSDictionary<NSString *, id> *)resources NS_DESIGNATED_INITIALIZER;

/*! @abstract A Boolean value indicating whether the extension background content is a service worker. */
@property (readonly, nonatomic) BOOL _hasServiceWorkerBackgroundContent;

/*! @abstract A Boolean value indicating whether the extension use modules for the background content. */
@property (readonly, nonatomic) BOOL _hasModularBackgroundContent;

/*! @abstract A Boolean value indicating whether the extension has a sidebar. */
@property (readonly, nonatomic) BOOL _hasSidebar;

@end

WK_HEADER_AUDIT_END(nullability, sendability)
