/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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

@class DOMElement;
@class DOMHTMLFormElement;
@class DOMHTMLInputElement;
@class DOMHTMLTextAreaElement;
#if TARGET_OS_IPHONE
@class DOMNode;
#endif
@class WebFrame;

/*!
    @protocol  WebFormSubmissionListener
*/
@protocol WebFormSubmissionListener <NSObject>
- (void)continue;
@end

/*!
    @protocol  WebFormDelegate
*/
WEBKIT_DEPRECATED_MAC(10_3, 10_14)
@protocol WebFormDelegate <NSObject>

// Various methods send by controls that edit text to their delegates, which are all
// analogous to similar methods in AppKit/NSControl.h.
// These methods are forwarded from widgets used in forms to the WebFormDelegate.

- (void)textFieldDidBeginEditing:(DOMHTMLInputElement *)element inFrame:(WebFrame *)frame;
- (void)textFieldDidEndEditing:(DOMHTMLInputElement *)element inFrame:(WebFrame *)frame;
- (void)textDidChangeInTextField:(DOMHTMLInputElement *)element inFrame:(WebFrame *)frame;
- (void)textDidChangeInTextArea:(DOMHTMLTextAreaElement *)element inFrame:(WebFrame *)frame;
- (void)didFocusTextField:(DOMHTMLInputElement *)element inFrame:(WebFrame *)frame;

- (BOOL)textField:(DOMHTMLInputElement *)element doCommandBySelector:(SEL)commandSelector inFrame:(WebFrame *)frame;
#if !TARGET_OS_IPHONE
- (BOOL)textField:(DOMHTMLInputElement *)element shouldHandleEvent:(NSEvent *)event inFrame:(WebFrame *)frame;
#endif
// Sent when a form is just about to be submitted (before the load is started)
// listener must be sent continue when the delegate is done.
- (void)frame:(WebFrame *)frame sourceFrame:(WebFrame *)sourceFrame willSubmitForm:(DOMElement *)form
    withValues:(NSDictionary *)values submissionListener:(id <WebFormSubmissionListener>)listener;

- (void)willSendSubmitEventToForm:(DOMHTMLFormElement *)element inFrame:(WebFrame *)sourceFrame withValues:(NSDictionary *)values;

@end

/*!
    @class WebFormDelegate
    @discussion The WebFormDelegate class responds to all WebFormDelegate protocol
    methods by doing nothing. It's provided for the convenience of clients who only want
    to implement some of the above methods and ignore others.
*/
@interface WebFormDelegate : NSObject <WebFormDelegate>
@end
