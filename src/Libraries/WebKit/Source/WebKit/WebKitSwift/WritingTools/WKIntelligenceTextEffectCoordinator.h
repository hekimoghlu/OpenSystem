/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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

#if !TARGET_OS_WATCH && !TARGET_OS_TV && __has_include(<WritingTools/WritingTools.h>)

#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif

#import <WebKit/_WKTextPreview.h>

NS_ASSUME_NONNULL_BEGIN

@protocol WKIntelligenceTextEffectCoordinating;

@class WTTextSuggestion;

NS_SWIFT_UI_ACTOR
@protocol WKIntelligenceTextEffectCoordinatorDelegate <NSObject>

#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
- (UIView *)viewForIntelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator;
#else
- (NSView *)viewForIntelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator;
#endif

#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
- (void)intelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator textPreviewsForRange:(NSRange)range completion:(void (^)(UITargetedPreview *))completion;
#else
- (void)intelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator textPreviewsForRange:(NSRange)range completion:(void (^)(NSArray<_WKTextPreview *> *))completion;
#endif

- (void)intelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator contentPreviewForRange:(NSRange)range completion:(void (^)(_WKTextPreview *))completion;

- (void)intelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator rectsForProofreadingSuggestionsInRange:(NSRange)range completion:(void (^)(NSArray<NSValue *> *))completion;

- (void)intelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator updateTextVisibilityForRange:(NSRange)range visible:(BOOL)visible identifier:(NSUUID *)identifier completion:(void (^)(void))completion;

- (void)intelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator decorateReplacementsForRange:(NSRange)range completion:(void (^)(void))completion;

- (void)intelligenceTextEffectCoordinator:(id<WKIntelligenceTextEffectCoordinating>)coordinator setSelectionForRange:(NSRange)range completion:(void (^)(void))completion;

@end

NS_SWIFT_UI_ACTOR
@protocol WKIntelligenceTextEffectCoordinating

@property (nonatomic, readonly) BOOL hasActiveEffects;

- (instancetype)initWithDelegate:(id<WKIntelligenceTextEffectCoordinatorDelegate>)delegate;

- (void)startAnimationForRange:(NSRange)range completion:(void (^)(void))completion;

- (void)requestReplacementWithProcessedRange:(NSRange)range finished:(BOOL)finished characterDelta:(NSInteger)characterDelta operation:(void (^)(void (^)(void)))operation completion:(void (^)(void))completion;

- (void)flushReplacementsWithCompletion:(void (^)(void))completion;

- (void)restoreSelectionAcceptedReplacements:(BOOL)acceptedReplacements completion:(void (^)(void))completion;

- (void)hideEffectsWithCompletion:(void (^)(void))completion;

- (void)showEffectsWithCompletion:(void (^)(void))completion;

@end

NS_ASSUME_NONNULL_END

#endif
