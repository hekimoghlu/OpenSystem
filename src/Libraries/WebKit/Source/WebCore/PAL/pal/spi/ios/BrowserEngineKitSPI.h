/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#pragma once

#if USE(BROWSERENGINEKIT)

#import <BrowserEngineKit/BrowserEngineKit.h>

#if USE(APPLE_INTERNAL_SDK)
// Note: SPI usage should be limited to testing purposes and binary compatibility with clients
// of existing WebKit SPI.
#import <BrowserEngineKit/BrowserEngineKit_Private.h>
#else

@class NSTextAlternatives;
@class UIKeyEvent;
@class UITextSuggestion;
@class UIWKDocumentContext;

@interface BEKeyEntry (ForTesting)
- (UIKeyEvent *)_uikitKeyEvent;
- (instancetype)_initWithUIKitKeyEvent:(UIKeyEvent *)keyEvent;
@end

@interface BETextAlternatives ()
@property (readonly) BOOL isLowConfidence;
- (NSTextAlternatives *)_nsTextAlternative;
- (instancetype)_initWithNSTextAlternatives:(NSTextAlternatives *)nsTextAlternatives;
@end

@interface BETextDocumentContext ()
@property (strong, nonatomic, readonly) UIWKDocumentContext *_uikitDocumentContext;
@property (nonatomic, copy) NSAttributedString *annotatedText;
@end

@interface BETextDocumentRequest ()
@property (nonatomic, assign) CGRect _documentRect;
@end

@interface BETextSuggestion ()
@property (nonatomic, readonly, strong) UITextSuggestion *_uikitTextSuggestion;
- (instancetype)_initWithUIKitTextSuggestion:(UITextSuggestion *)suggestion;
@end

#endif

#endif // USE(BROWSERENGINEKIT)
