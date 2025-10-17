/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#if PLATFORM(MAC)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSTextChecker.h>

#else

extern NSString *NSTextCheckingInsertionPointKey;
extern NSString *NSTextCheckingSuppressInitialCapitalizationKey;
#if HAVE(INLINE_PREDICTIONS)
extern NSString *NSTextCompletionAttributeName;
#endif

@interface NSSpellChecker ()

#if HAVE(INLINE_PREDICTIONS)
@property (class, readonly, getter=isAutomaticInlineCompletionEnabled) BOOL automaticInlineCompletionEnabled;
- (NSTextCheckingResult *)completionCandidateFromCandidates:(NSArray<NSTextCheckingResult *> *)candidates;
- (void)showCompletionForCandidate:(NSTextCheckingResult *)candidate selectedRange:(NSRange)selectedRange offset:(NSUInteger)offset inString:(NSString *)string rect:(NSRect)rect view:(NSView *)view completionHandler:(void (^)(NSDictionary *resultDictionary))completionBlock;
- (void)showCompletionForCandidate:(NSTextCheckingResult *)candidate selectedRange:(NSRange)selectedRange offset:(NSUInteger)offset inString:(NSString *)string rect:(NSRect)rect view:(NSView *)view client:(id <NSTextInputClient>)client completionHandler:(void (^)(NSDictionary *resultDictionary))completionBlock;
#endif

- (NSString *)languageForWordRange:(NSRange)range inString:(NSString *)string orthography:(NSOrthography *)orthography;
- (BOOL)deletesAutospaceBeforeString:(NSString *)string language:(NSString *)language;
- (void)_preflightChosenSpellServer;

+ (BOOL)grammarCheckingEnabled;

@end

#if HAVE(INLINE_PREDICTIONS)
typedef NS_OPTIONS(uint64_t, NSTextCheckingTypeAppKitTemporary) {
#if HAVE(NS_TEXT_CHECKING_TYPE_MATH_COMPLETION)
    _NSTextCheckingTypeMathCompletion   = 1ULL << 28,
#endif
    _NSTextCheckingTypeSingleCompletion = 1ULL << 29,
};
#endif

#endif // USE(APPLE_INTERNAL_SDK)

#if HAVE(AUTOCORRECTION_ENHANCEMENTS)
// FIXME: rdar://105853874 Remove staging code.
@interface NSSpellChecker (Staging_105286196)
+ (NSColor *)correctionIndicatorUnderlineColor;
@end
#endif

#endif // PLATFORM(MAC)
