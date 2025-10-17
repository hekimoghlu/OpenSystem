/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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

@class WebFrame;

/*!
    This informal protocol enables a plug-in to request that its containing application
    perform certain operations.
*/

@interface NSObject (WebPlugInContainer)

/*!
    @method webPlugInContainerLoadRequest:inFrame:
    @abstract Tell the application to show a URL in a target frame
    @param request The request to be loaded.
    @param target The target frame. If the frame with the specified target is not
    found, a new window is opened and the main frame of the new window is named
    with the specified target.  If nil is specified, the frame that contains
    the applet is targeted.
*/
- (void)webPlugInContainerLoadRequest:(NSURLRequest *)request inFrame:(NSString *)target;

/*!
    @method webPlugInContainerShowStatus:
    @abstract Tell the application to show the specified status message.
    @param message The string to be shown.
*/
- (void)webPlugInContainerShowStatus:(NSString *)message;

#if !TARGET_OS_IPHONE
/*!
    @property webPlugInContainerSelectionColor
    @abstract The color that should be used for any special drawing when
    plug-in is selected.
*/
@property (nonatomic, readonly, strong) NSColor *webPlugInContainerSelectionColor;
#endif

/*!
    @property webFrame
    @abstract Allows the plug-in to access the WebFrame that
    contains the plug-in.  This method will not be implemented by containers that 
    are not WebKit based.
*/
@property (nonatomic, readonly, strong) WebFrame *webFrame;

@end
