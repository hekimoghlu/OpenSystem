/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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
#import <WebKitLegacy/DOMDocument.h>

@class DOMHTMLHeadElement;
@class DOMHTMLScriptElement;

@interface DOMDocument (DOMDocumentPrivate)
@property (readonly, copy) NSString *contentType;
@property (copy) NSString *dir;
@property (readonly, strong) DOMHTMLHeadElement *head;
@property (readonly, copy) NSString *compatMode;
#if !TARGET_OS_IPHONE
@property (readonly) BOOL webkitIsFullScreen;
@property (readonly) BOOL webkitFullScreenKeyboardInputAllowed;
@property (readonly, strong) DOMElement *webkitCurrentFullScreenElement;
@property (readonly) BOOL webkitFullscreenEnabled;
@property (readonly, strong) DOMElement *webkitFullscreenElement;
#endif
@property (readonly, copy) NSString *visibilityState;
@property (readonly) BOOL hidden;
@property (readonly, strong) DOMHTMLScriptElement *currentScript;
@property (readonly, copy) NSString *origin;
@property (readonly, strong) DOMElement *scrollingElement;
@property (readonly, strong) DOMHTMLCollection *children;
@property (readonly, strong) DOMElement *firstElementChild;
@property (readonly, strong) DOMElement *lastElementChild;
@property (readonly) unsigned childElementCount;

- (DOMRange *)caretRangeFromPoint:(int)x y:(int)y;
#if !TARGET_OS_IPHONE
- (void)webkitExitFullscreen;
#endif
@end
