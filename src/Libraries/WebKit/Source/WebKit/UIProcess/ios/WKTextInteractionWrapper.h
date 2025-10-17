/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

@class WKContentView;

@interface WKTextInteractionWrapper : NSObject

- (instancetype)initWithView:(WKContentView *)contentView;

- (void)activateSelection;
- (void)deactivateSelection;
- (void)didEndScrollingOverflow;
- (void)selectionChanged;
- (void)setGestureRecognizers;
- (void)willStartScrollingOverflow:(UIScrollView *)scrollView;
- (void)selectionChangedWithGestureAt:(CGPoint)point withGesture:(WKBEGestureType)gestureType withState:(UIGestureRecognizerState)gestureState withFlags:(WKBESelectionFlags)flags;
- (void)selectionChangedWithTouchAt:(CGPoint)point withSelectionTouch:(WKBESelectionTouchPhase)touch withFlags:(WKBESelectionFlags)flags;
- (void)lookup:(NSString *)textWithContext withRange:(NSRange)range fromRect:(CGRect)presentationRect;
- (void)showShareSheetFor:(NSString *)selectedTerm fromRect:(CGRect)presentationRect;
- (void)showTextServiceFor:(NSString *)selectedTerm fromRect:(CGRect)presentationRect;
- (void)scheduleReplacementsForText:(NSString *)text;
- (void)scheduleChineseTransliterationForText:(NSString *)text;
- (void)willStartScrollingOrZooming;
- (void)didEndScrollingOrZooming;
- (void)selectWord;
- (void)selectAll:(id)sender;
- (void)translate:(NSString *)text fromRect:(CGRect)presentationRect;
- (void)prepareToMoveSelectionContainer:(UIView *)newContainer;
- (void)setNeedsSelectionUpdate;

- (void)willBeginDragLift;
- (void)didConcludeDrop;

- (void)reset;

#if USE(UICONTEXTMENU)
- (void)setExternalContextMenuInteractionDelegate:(id<UIContextMenuInteractionDelegate>)delegate;
@property (nonatomic, strong, readonly) UIContextMenuInteraction *contextMenuInteraction;
#endif

@property (nonatomic, readonly) NSArray<UIView *> *managedTextSelectionViews;
@property (nonatomic, readonly) UIWKTextInteractionAssistant *textInteractionAssistant;

@end

#endif // PLATFORM(IOS_FAMILY)
