/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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

#import <WebKit/WKWebProcessPlugInBrowserContextController.h>
#import <WebKit/WKWebProcessPlugInFrame.h>
#import <WebKit/WKWebProcessPlugInNodeHandle.h>

@protocol WKWebProcessPlugInFormDelegatePrivate <NSObject>

@optional

- (void)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller didFocusTextField:(WKWebProcessPlugInNodeHandle *)textField inFrame:(WKWebProcessPlugInFrame *)frame;
- (void)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller willSendSubmitEventToForm:(WKWebProcessPlugInNodeHandle *)form inFrame:(WKWebProcessPlugInFrame *)sourceFrame targetFrame:(WKWebProcessPlugInFrame *)targetFrame values:(NSDictionary *)values;
- (void)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller textDidChangeInTextField:(WKWebProcessPlugInNodeHandle *)textField inFrame:(WKWebProcessPlugInFrame *)frame initiatedByUserTyping:(BOOL)initiatedByUserTyping;

// The return value is exposed in the UI process via the userObject parameter to -[id <_WKInputDelegate> _webView:willSubmitFormValues:userObject:submissionHandler:].
- (NSObject <NSSecureCoding> *)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller willSubmitForm:(WKWebProcessPlugInNodeHandle *)form toFrame:(WKWebProcessPlugInFrame *)frame fromFrame:(WKWebProcessPlugInFrame *)sourceFrame withValues:(NSDictionary *)values;

// The return value is exposed in the UI process via the userObject property of the _WKFormInputSession object.
- (NSObject <NSSecureCoding> *)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller willBeginInputSessionForElement:(WKWebProcessPlugInNodeHandle *)node inFrame:(WKWebProcessPlugInFrame *)frame;
- (NSObject <NSSecureCoding> *)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller willBeginInputSessionForElement:(WKWebProcessPlugInNodeHandle *)node inFrame:(WKWebProcessPlugInFrame *)frame userIsInteracting:(BOOL)userIsInteracting WK_API_AVAILABLE(ios(10.0));

- (BOOL)_webProcessPlugInBrowserContextControllerShouldNotifyOnFormChanges:(WKWebProcessPlugInBrowserContextController *)controller;
- (void)_webProcessPlugInBrowserContextController:(WKWebProcessPlugInBrowserContextController *)controller didAssociateFormControls:(NSArray *)elements;

@end
