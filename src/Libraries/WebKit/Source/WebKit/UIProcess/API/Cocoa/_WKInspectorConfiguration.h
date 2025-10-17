/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

@class WKProcessPool;
@protocol WKURLSchemeHandler;

WK_CLASS_AVAILABLE(macos(12.0))
@interface _WKInspectorConfiguration : NSObject <NSCopying>
/**
 * @abstract Sets the URL scheme handler object for the given URL scheme.
 * @param urlSchemeHandler The object to register.
 * @param scheme The URL scheme the object will handle.
 * @discussion This is used to register additional schemes for loading resources in the inspector page,
 * such as to register schemes used by extensions. This method has the same behavior and restrictions as
 * described in the documentation of -[WKWebView setURLSchemeHandler:forURLScheme:].
 */
- (void)setURLSchemeHandler:(id <WKURLSchemeHandler>)urlSchemeHandler forURLScheme:(NSString *)urlScheme;

/*! @abstract The process pool from which to obtain web content processes on behalf of Web Inspector.
 @discussion When a Web Inspector instance opens, a new web content process
 will be created for it from the specified pool, or an existing process in that pool will be used.
 This process pool is also used to obtain web content processes for tabs created via _WKInspectorExtension.
 If unspecified, all Web Inspector instances will use a private process pool that is separate from web content.
*/
@property (nonatomic, strong, nullable) WKProcessPool *processPool;

/*! @abstract An identifier that is set for all pages associated with this inspector configuration.
 @discussion This can be used to uniquely identify Web Inspector-related pages from within the injected bundle.
 If unspecified, all Web Inspector instances will use a private group identifier that is separate from web content.
*/
@property (nonatomic, copy, nullable) NSString *groupIdentifier;
@end

NS_ASSUME_NONNULL_END
