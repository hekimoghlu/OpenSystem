/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#import <WebKitLegacy/WebDOMOperations.h>
#import <JavaScriptCore/JSBase.h>

#if TARGET_OS_IPHONE
#import <WebKitLegacy/WAKAppKitStubs.h>
#else
#import <AppKit/NSEvent.h>
#import <WebKitLegacy/DOMWheelEvent.h>
#endif

@interface DOMElement (WebDOMElementOperationsPrivate)
+ (DOMElement *)_DOMElementFromJSContext:(JSContextRef)context value:(JSValueRef)value;
@end

@interface DOMHTMLInputElement (WebDOMHTMLInputElementOperationsPrivate)
- (BOOL)_isAutofilled;
- (void)_setAutofilled:(BOOL)autofilled;

- (BOOL)_isAutoFilledAndViewable;
- (void)_setAutoFilledAndViewable:(BOOL)autoFilledAndViewable;
@end

@interface DOMNode (WebDOMNodeOperationsPendingPublic)
- (NSString *)markupString;
- (NSRect)_renderRect:(bool *)isReplaced;
@end

typedef BOOL (^WebArchiveSubframeFilter)(WebFrame* subframe);

@interface DOMNode (WebDOMNodeOperationsPrivate)
- (WebArchive *)webArchiveByFilteringSubframes:(WebArchiveSubframeFilter)webArchiveSubframeFilter;
#if TARGET_OS_IPHONE
- (BOOL)isHorizontalWritingMode;
- (void)hidePlaceholder;
- (void)showPlaceholderIfNecessary;
#endif
@end

#if !TARGET_OS_IPHONE
@interface DOMWheelEvent (WebDOMWheelEventOperationsPrivate)
- (NSEventPhase)_phase;
- (NSEventPhase)_momentumPhase;
@end
#endif
