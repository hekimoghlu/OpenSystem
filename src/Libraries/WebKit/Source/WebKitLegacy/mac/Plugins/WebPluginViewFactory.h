/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#import <WebKitLegacy/WebKitAvailability.h>

#if !TARGET_OS_IPHONE
#import <AppKit/AppKit.h>
#endif

/*!
    @constant WebPlugInBaseURLKey REQUIRED. The base URL of the document containing
    the plug-in's view.
*/
extern NSString *WebPlugInBaseURLKey WEBKIT_DEPRECATED_MAC(10_3, 10_14);

/*!
    @constant WebPlugInAttributesKey REQUIRED. The dictionary containing the names
    and values of all attributes of the HTML element associated with the plug-in AND
    the names and values of all parameters to be passed to the plug-in (e.g. PARAM
    elements within an APPLET element). In the case of a conflict between names,
    the attributes of an element take precedence over any PARAMs.  All of the keys
    and values in this NSDictionary must be NSStrings.
*/
extern NSString *WebPlugInAttributesKey WEBKIT_DEPRECATED_MAC(10_3, 10_14);

/*!
    @constant WebPlugInContainer OPTIONAL. An object that conforms to the
    WebPlugInContainer informal protocol. This object is used for
    callbacks from the plug-in to the app. if this argument is nil, no callbacks will
    occur.
*/
extern NSString *WebPlugInContainerKey WEBKIT_DEPRECATED_MAC(10_3, 10_14);

/*!
    @constant WebPlugInContainingElementKey The DOMElement that was used to specify
    the plug-in.  May be nil.
*/
extern NSString *WebPlugInContainingElementKey WEBKIT_DEPRECATED_MAC(10_3, 10_14);

/*!
 @constant WebPlugInShouldLoadMainResourceKey REQUIRED. NSNumber (BOOL) indicating whether the plug-in should load its
 own main resource (the "src" URL, in most cases). If YES, the plug-in should load its own main resource. If NO, the
 plug-in should use the data provided by WebKit. See -webPlugInMainResourceReceivedData: in WebPluginPrivate.h.
 For compatibility with older versions of WebKit, the plug-in should assume that the value for
 WebPlugInShouldLoadMainResourceKey is NO if it is absent from the arguments dictionary.
 */
extern NSString *WebPlugInShouldLoadMainResourceKey WEBKIT_DEPRECATED_MAC(10_6, 10_14);

/*!
    @protocol WebPlugInViewFactory
    @discussion WebPlugInViewFactory are used to create the NSView for a plug-in.
    The principal class of the plug-in bundle must implement this protocol.
*/

WEBKIT_DEPRECATED_MAC(10_3, 10_14)
@protocol WebPlugInViewFactory <NSObject>

/*!
    @method plugInViewWithArguments: 
    @param arguments The arguments dictionary with the mentioned keys and objects. This method is required to implement.
    @result Returns an NSView object that conforms to the WebPlugIn informal protocol.
*/
#if !TARGET_OS_IPHONE
+ (NSView *)plugInViewWithArguments:(NSDictionary *)arguments;
#else
// +plugInViewWithArguments: returns a UIView subclass
+ (id)plugInViewWithArguments:(NSDictionary *)arguments;
#endif

@end
