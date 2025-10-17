/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKBrowserEngineDefinitions.h"
#import <pal/spi/ios/BrowserEngineKitSPI.h>

@interface WKExtendedTextInputTraits : NSObject
#if USE(BROWSERENGINEKIT)
    <BEExtendedTextInputTraits>
#endif

@property (nonatomic) UITextAutocapitalizationType autocapitalizationType;
@property (nonatomic) UITextAutocorrectionType autocorrectionType;
@property (nonatomic) UITextSpellCheckingType spellCheckingType;
@property (nonatomic) UITextSmartQuotesType smartQuotesType;
@property (nonatomic) UITextSmartDashesType smartDashesType;
#if HAVE(INLINE_PREDICTIONS)
@property (nonatomic) UITextInlinePredictionType inlinePredictionType;
#endif
@property (nonatomic) UIKeyboardType keyboardType;
@property (nonatomic) UIKeyboardAppearance keyboardAppearance;
@property (nonatomic) UIReturnKeyType returnKeyType;
@property (nonatomic, getter=isSecureTextEntry) BOOL secureTextEntry;
@property (nonatomic, getter=isSingleLineDocument) BOOL singleLineDocument;
@property (nonatomic, getter=isTypingAdaptationEnabled) BOOL typingAdaptationEnabled;
@property (nonatomic, copy) UITextContentType textContentType;
@property (nonatomic, copy) UITextInputPasswordRules *passwordRules;
@property (nonatomic) UITextSmartInsertDeleteType smartInsertDeleteType;
@property (nonatomic) BOOL enablesReturnKeyAutomatically;

@property (nonatomic, strong) UIColor *insertionPointColor;
@property (nonatomic, strong) UIColor *selectionHandleColor;
@property (nonatomic, strong) UIColor *selectionHighlightColor;

- (void)setSelectionColorsToMatchTintColor:(UIColor *)tintColor;
- (void)restoreDefaultValues;

@end

#endif // PLATFORM(IOS_FAMILY)
