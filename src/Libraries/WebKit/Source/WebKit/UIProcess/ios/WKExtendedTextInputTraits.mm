/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 24, 2025.
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
#import "config.h"
#import "WKExtendedTextInputTraits.h"

#if PLATFORM(IOS_FAMILY)

#import <wtf/RetainPtr.h>

@implementation WKExtendedTextInputTraits {
    RetainPtr<UITextContentType> _textContentType;
    RetainPtr<UIColor> _insertionPointColor;
    RetainPtr<UIColor> _selectionHandleColor;
    RetainPtr<UIColor> _selectionHighlightColor;
    RetainPtr<UITextInputPasswordRules> _passwordRules;
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    self.typingAdaptationEnabled = YES;
    self.autocapitalizationType = UITextAutocapitalizationTypeSentences;
    return self;
}

- (void)setPasswordRules:(UITextInputPasswordRules *)rules
{
    _passwordRules = adoptNS(rules.copy);
}

- (UITextInputPasswordRules *)passwordRules
{
    return adoptNS([_passwordRules copy]).autorelease();
}

- (void)setTextContentType:(UITextContentType)type
{
    _textContentType = adoptNS(type.copy);
}

- (UITextContentType)textContentType
{
    return adoptNS([_textContentType copy]).autorelease();
}

- (void)setInsertionPointColor:(UIColor *)color
{
    _insertionPointColor = color;
}

- (UIColor *)insertionPointColor
{
    return _insertionPointColor.get();
}

- (void)setSelectionHandleColor:(UIColor *)color
{
    _selectionHandleColor = color;
}

- (UIColor *)selectionHandleColor
{
    return _selectionHandleColor.get();
}

- (void)setSelectionHighlightColor:(UIColor *)color
{
    _selectionHighlightColor = color;
}

- (UIColor *)selectionHighlightColor
{
    return _selectionHighlightColor.get();
}

- (void)setSelectionColorsToMatchTintColor:(UIColor *)tintColor
{
    static constexpr auto selectionHighlightAlphaComponent = 0.2;
    BOOL shouldUseTintColor = tintColor && tintColor != UIColor.systemBlueColor;
    self.insertionPointColor = shouldUseTintColor ? tintColor : nil;
    self.selectionHandleColor = shouldUseTintColor ? tintColor : nil;
    self.selectionHighlightColor = shouldUseTintColor ? [tintColor colorWithAlphaComponent:selectionHighlightAlphaComponent] : nil;
}

- (void)restoreDefaultValues
{
    self.typingAdaptationEnabled = YES;
#if HAVE(INLINE_PREDICTIONS)
    self.inlinePredictionType = UITextInlinePredictionTypeDefault;
#endif
    self.autocapitalizationType = UITextAutocapitalizationTypeSentences;
    self.autocorrectionType = UITextAutocorrectionTypeDefault;
    self.spellCheckingType = UITextSpellCheckingTypeDefault;
    self.smartQuotesType = UITextSmartQuotesTypeDefault;
    self.smartDashesType = UITextSmartDashesTypeDefault;
    self.keyboardType = UIKeyboardTypeDefault;
    self.keyboardAppearance = UIKeyboardAppearanceDefault;
    self.returnKeyType = UIReturnKeyDefault;
    self.secureTextEntry = NO;
    self.singleLineDocument = NO;
    self.textContentType = nil;
    self.passwordRules = nil;
    self.smartInsertDeleteType = UITextSmartInsertDeleteTypeDefault;
    self.enablesReturnKeyAutomatically = NO;
    self.insertionPointColor = nil;
    self.selectionHandleColor = nil;
    self.selectionHighlightColor = nil;
}

@end

#endif // PLATFORM(IOS_FAMILY)
