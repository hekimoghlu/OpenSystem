/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#import <WebKit/WKWebView.h>
#import <WebKit/_WKElementAction.h>
#import <WebKit/_WKFrameHandle.h>
#import <WebKit/_WKTapHandlingResult.h>

#if TARGET_OS_IPHONE

@class _WKTextInputContext;
@class UIEventAttribution;
@class UIGestureRecognizer;
@class BEDocumentContext;
@class UIWKDocumentRequest;
@class UITapGestureRecognizer;

@interface WKWebView (WKTestingIOS)

@property (nonatomic, readonly) NSString *textContentTypeForTesting;
@property (nonatomic, readonly) NSString *selectFormPopoverTitle;
@property (nonatomic, readonly) NSString *formInputLabel;
@property (nonatomic, readonly) CGRect _inputViewBoundsInWindow;
@property (nonatomic, readonly) NSString *_uiViewTreeAsText;
@property (nonatomic, readonly) NSNumber *_stableStateOverride;
@property (nonatomic, readonly) CGRect _dragCaretRect;
@property (nonatomic, readonly, getter=_isAnimatingDragCancel) BOOL _animatingDragCancel;
@property (nonatomic, readonly) CGRect _tapHighlightViewRect;
@property (nonatomic, readonly) UIGestureRecognizer *_imageAnalysisGestureRecognizer;
@property (nonatomic, readonly) UITapGestureRecognizer *_singleTapGestureRecognizer;
@property (nonatomic, readonly, getter=_isKeyboardScrollingAnimationRunning) BOOL _keyboardScrollingAnimationRunning;

- (void)keyboardAccessoryBarNext;
- (void)keyboardAccessoryBarPrevious;
- (void)dismissFormAccessoryView;
- (NSArray<NSString *> *)_filePickerAcceptedTypeIdentifiers;
- (void)_dismissFilePicker;
- (void)selectFormAccessoryPickerRow:(int)rowIndex;
- (BOOL)selectFormAccessoryHasCheckedItemAtRow:(long)rowIndex;
- (void)setSelectedColorForColorPicker:(UIColor *)color;
- (void)_selectDataListOption:(int)optionIndex;
- (BOOL)_isShowingDataListSuggestions;
- (void)selectWordBackwardForTesting;

- (BOOL)_mayContainEditableElementsInRect:(CGRect)rect;
- (void)_requestTextInputContextsInRect:(CGRect)rect completionHandler:(void (^)(NSArray<_WKTextInputContext *> *))completionHandler;
- (void)_focusTextInputContext:(_WKTextInputContext *)context placeCaretAt:(CGPoint)point completionHandler:(void (^)(UIResponder<UITextInput> *))completionHandler;
- (void)_willBeginTextInteractionInTextInputContext:(_WKTextInputContext *)context;
- (void)_didFinishTextInteractionInTextInputContext:(_WKTextInputContext *)context;
- (void)setTimePickerValueToHour:(NSInteger)hour minute:(NSInteger)minute;
- (double)timePickerValueHour;
- (double)timePickerValueMinute;

- (NSDictionary *)_propertiesOfLayerWithID:(unsigned long long)layerID;
- (void)_simulateElementAction:(_WKElementActionType)actionType atLocation:(CGPoint)location;
- (void)_simulateLongPressActionAtLocation:(CGPoint)location;
- (void)_simulateTextEntered:(NSString *)text;

- (void)_doAfterReceivingEditDragSnapshotForTesting:(dispatch_block_t)action;

- (void)_triggerSystemPreviewActionOnElement:(uint64_t)elementID document:(NSString*)documentID page:(uint64_t)pageID;

- (void)_setDeviceOrientationUserPermissionHandlerForTesting:(BOOL (^)(void))handler;

- (void)_setDeviceHasAGXCompilerServiceForTesting;

- (void)_resetObscuredInsetsForTesting;
- (BOOL)_hasResizeAssertion;
- (void)_simulateSelectionStart;

+ (void)_resetPresentLockdownModeMessage;

- (void)_doAfterNextVisibleContentRectAndStablePresentationUpdate:(void (^)(void))updateBlock;

- (NSString *)_scrollbarState:(unsigned long long)scrollingNodeID processID: (unsigned long long)processID isVertical:(bool)isVertical;

@end

#endif // TARGET_OS_IPHONE
