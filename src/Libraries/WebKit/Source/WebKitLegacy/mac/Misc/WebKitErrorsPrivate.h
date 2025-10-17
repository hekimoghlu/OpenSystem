/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#import <WebKitLegacy/WebKitErrors.h>

#define WebKitErrorPlugInCancelledConnection 203
// FIXME: WebKitErrorPlugInWillHandleLoad is used for the cancel we do to prevent loading plugin content twice.  See <rdar://problem/4258008>
#define WebKitErrorPlugInWillHandleLoad 204

/*!
    @enum
    @abstract Policy errors - Pending Public API Review
    @constant WebKitErrorCannotUseRestrictedPort
    @constant WebKitErrorFrameLoadBlockedByContentFilter
*/
enum {
    WebKitErrorCannotUseRestrictedPort =                        103,
    WebKitErrorFrameLoadBlockedByContentFilter =                105,
};

/*!
    @enum
    @abstract Geolocation errors
    @constant WebKitErrorGeolocationLocationUnknown
*/
enum {
    WebKitErrorGeolocationLocationUnknown  =                    300,
};

@interface NSError (WebKitExtras)
+ (NSError *)_webKitErrorWithCode:(int)code failingURL:(NSString *)URL;
+ (NSError *)_webKitErrorWithDomain:(NSString *)domain code:(int)code URL:(NSURL *)URL;

- (id)_initWithPluginErrorCode:(int)code
                    contentURL:(NSURL *)contentURL
                 pluginPageURL:(NSURL *)pluginPageURL
                    pluginName:(NSString *)pluginName
                      MIMEType:(NSString *)MIMEType;
@end
