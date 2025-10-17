/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#import <WebKitLegacy/WebArchive.h>
#import <WebKitLegacy/WebResource.h>

// This makes it possible to use -[NSBundle classNamed:] to find classes from WebKitLegacy.framework.

#define DEFINE_BUNDLE_FOR_CLASS(className) \
@interface className (WKFoundationSupport) \
@end \
@implementation className (WKFoundationSupport) \
+ (NSBundle *)bundleForClass \
{ \
    return [NSBundle bundleForClass:NSClassFromString(@"WKWebView")]; \
} \
@end

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
DEFINE_BUNDLE_FOR_CLASS(WebArchive)
DEFINE_BUNDLE_FOR_CLASS(WebResource)
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
