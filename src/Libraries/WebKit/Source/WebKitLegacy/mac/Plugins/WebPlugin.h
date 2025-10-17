/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <WebKitLegacy/WebKitAvailability.h>

#if !TARGET_OS_IPHONE
#import <AppKit/AppKit.h>
#endif

/*!
    WebPlugIn is an informal protocol that enables interaction between an application
    and web related plug-ins it may contain.
*/

@interface NSObject (WebPlugIn)

/*!
    @method webPlugInInitialize
    @abstract Tell the plug-in to perform one-time initialization.
    @discussion This method must be only called once per instance of the plug-in
    object and must be called before any other methods in this protocol.
*/
- (void)webPlugInInitialize;

/*!
    @method webPlugInStart
    @abstract Tell the plug-in to start normal operation.
    @discussion The plug-in usually begins drawing, playing sounds and/or
    animation in this method.  This method must be called before calling webPlugInStop.
    This method may called more than once, provided that the application has
    already called webPlugInInitialize and that each call to webPlugInStart is followed
    by a call to webPlugInStop.
*/
- (void)webPlugInStart;

/*!
    @method webPlugInStop
    @abstract Tell the plug-in to stop normal operation.
    @discussion webPlugInStop must be called before this method.  This method may be
    called more than once, provided that the application has already called
    webPlugInInitialize and that each call to webPlugInStop is preceded by a call to
    webPlugInStart.
*/
- (void)webPlugInStop;

/*!
    @method webPlugInDestroy
    @abstract Tell the plug-in perform cleanup and prepare to be deallocated.
    @discussion The plug-in typically releases memory and other resources in this
    method.  If the plug-in has retained the WebPlugInContainer, it must release
    it in this mehthod.  This method must be only called once per instance of the
    plug-in object.  No other methods in this interface may be called after the
    application has called webPlugInDestroy.
*/
- (void)webPlugInDestroy;

#if !TARGET_OS_IPHONE
/*!
    @method webPlugInSetIsSelected:
    @discusssion Informs the plug-in whether or not it is selected.  This is typically
    used to allow the plug-in to alter it's appearance when selected.
*/
- (void)webPlugInSetIsSelected:(BOOL)isSelected;
#endif

/*!
    @property objectForWebScript
    @discussion objectForWebScript is used to expose a plug-in's scripting interface.  The 
    methods of the object are exposed to the script environment.  See the WebScripting
    informal protocol for more details.
    @result Returns the object that exposes the plug-in's interface.  The class of this
    object can implement methods from the WebScripting informal protocol.
*/
@property (nonatomic, readonly, strong) id objectForWebScript;

/*!
    @method webPlugInMainResourceDidReceiveResponse:
    @abstract Called on the plug-in when WebKit receives -connection:didReceiveResponse:
    for the plug-in's main resource.
    @discussion This method is only sent to the plug-in if the
    WebPlugInShouldLoadMainResourceKey argument passed to the plug-in was NO.
*/
- (void)webPlugInMainResourceDidReceiveResponse:(NSURLResponse *)response WEBKIT_AVAILABLE_MAC(10_6);

/*!
    @method webPlugInMainResourceDidReceiveData:
    @abstract Called on the plug-in when WebKit recieves -connection:didReceiveData:
    for the plug-in's main resource.
    @discussion This method is only sent to the plug-in if the
    WebPlugInShouldLoadMainResourceKey argument passed to the plug-in was NO.
*/
- (void)webPlugInMainResourceDidReceiveData:(NSData *)data WEBKIT_AVAILABLE_MAC(10_6);

/*!
    @method webPlugInMainResourceDidFailWithError:
    @abstract Called on the plug-in when WebKit receives -connection:didFailWithError:
    for the plug-in's main resource.
    @discussion This method is only sent to the plug-in if the
    WebPlugInShouldLoadMainResourceKey argument passed to the plug-in was NO.
*/
- (void)webPlugInMainResourceDidFailWithError:(NSError *)error WEBKIT_AVAILABLE_MAC(10_6);

/*!
    @method webPlugInMainResourceDidFinishLoading
    @abstract Called on the plug-in when WebKit receives -connectionDidFinishLoading:
    for the plug-in's main resource.
    @discussion This method is only sent to the plug-in if the
    WebPlugInShouldLoadMainResourceKey argument passed to the plug-in was NO.
*/
- (void)webPlugInMainResourceDidFinishLoading WEBKIT_AVAILABLE_MAC(10_6);

#if TARGET_OS_IPHONE

// FIXME: Comment me
- (Class)webPlugInFullScreenWindowClass;

// FIXME: Comment me
- (void)webPlugInWillEnterFullScreenWithFrame:(CGRect)newFrame;

// FIXME: Comment me
- (void)webPlugInWillLeaveFullScreenWithFrame:(CGRect)newFrame;

// FIXME: Comment me
- (BOOL)webPlugInReceivesEventsDirectly;

// FIXME: Comment me
- (void)webPlugInLayout;

// FIXME: Comment me
- (void)webPlugInDidDraw;

/*!
 @method webPlugInStopForPageCache
 @abstract Tell the plug-in to stop normal operation because the page the plug-in
 belongs to is entering a cache.
 @discussion A page in the PageCache can be quickly resumed. This is much like
 pausing and resuming a plug-in except the frame containing the plug-in will
 not be visible, is not active, and may even have been torn down. The API contract
 for messages before and after this message are the same as -webPlugInStop.
 */
- (void)webPlugInStopForPageCache;

#endif

@end
