/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
#import <WebKit/WKBase.h>
#import <WebKit/WKWebView.h>

#if !TARGET_OS_IPHONE

@class _WKFrameHandle;

@interface WKWebView (WKTestingMac)

@property (nonatomic, readonly) BOOL _hasActiveVideoForControlsManager;
@property (nonatomic, readonly) BOOL _shouldRequestCandidates;
@property (nonatomic, readonly) BOOL _allowsInlinePredictions;
@property (nonatomic, readonly) NSMenu *_activeMenu;

- (void)_requestControlledElementID;
- (void)_handleControlledElementIDResponse:(NSString *)identifier;

- (void)_handleAcceptedCandidate:(NSTextCheckingResult *)candidate;
- (void)_didHandleAcceptedCandidate;

- (void)_forceRequestCandidates;
- (void)_didUpdateCandidateListVisibility:(BOOL)visible;

- (void)_insertText:(id)string replacementRange:(NSRange)replacementRange;
- (NSRect)_candidateRect;

- (NSSet<NSView *> *)_pdfHUDs;

- (void)_retrieveAccessibilityTreeData:(void (^)(NSData *, NSError *))completionHandler;

- (void)_setSelectedColorForColorPicker:(NSColor *)color;

@property (nonatomic, readonly) BOOL _secureEventInputEnabledForTesting;

- (void)_createFlagsChangedEventMonitorForTesting;
- (BOOL)_hasFlagsChangedEventMonitorForTesting;

@end

#endif // !TARGET_OS_IPHONE
