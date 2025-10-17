/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#import <Foundation/Foundation.h>
#import <WebKit/WKBase.h>

@class WKWebProcessPlugInController;
@class WKWebProcessPlugInBrowserContextController;

@protocol WKWebProcessPlugIn <NSObject>
@optional
- (void)webProcessPlugIn:(WKWebProcessPlugInController *)plugInController initializeWithObject:(id)initializationObject;
- (void)webProcessPlugIn:(WKWebProcessPlugInController *)plugInController didCreateBrowserContextController:(WKWebProcessPlugInBrowserContextController *)browserContextController;
- (void)webProcessPlugIn:(WKWebProcessPlugInController *)plugInController willDestroyBrowserContextController:(WKWebProcessPlugInBrowserContextController *)browserContextController;
- (NSArray *)additionalClassesForParameterCoder;
@end

WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKWebProcessPlugInController : NSObject
- (void)extendClassesForParameterCoder:(NSArray *)classes WK_API_AVAILABLE(macos(10.14), ios(12.0));

@property (readonly) id parameters;

@end
