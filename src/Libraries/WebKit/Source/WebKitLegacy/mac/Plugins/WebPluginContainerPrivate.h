/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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

#if !TARGET_OS_IPHONE
#import <AppKit/AppKit.h>
#endif

@interface NSObject (WebPlugInContainerPrivate)

- (id)_webPluginContainerCheckIfAllowedToLoadRequest:(NSURLRequest *)Request inFrame:(NSString *)target resultObject:(id)obj selector:(SEL)selector;

- (void)_webPluginContainerCancelCheckIfAllowedToLoadRequest:(id)checkIdentifier;

#if TARGET_OS_IPHONE
// Call when the plug-in shows/hides its full-screen UI.
- (void)webPlugInContainerWillShowFullScreenForView:(id)plugInView;
- (void)webPlugInContainerDidHideFullScreenForView:(id)plugInView;

/*!
 @method processingUserGesture
 @discussion The processingUserGesture method allows the plug-in to find out if
 a user gesture is currently being processed. The plug-in may use this information
 to allow or deny certain actions.  This method will not be implemented by containers that
 are not WebKit based.
 @result Returns a boolean value, YES to indicate that a user gesture is being processed,
 NO otherwise.
 */
- (BOOL)processingUserGesture;
#endif

@end
