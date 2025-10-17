/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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

#if defined(__OBJC__) && __OBJC__
#import <pal/spi/ios/BrowserEngineKitSPI.h>
#endif

#if USE(BROWSERENGINEKIT)
// Scroll view
#define WKBEScrollView                                          BEScrollView
#define WKBEScrollViewDelegate                                  BEScrollViewDelegate
#define WKBEScrollViewScrollUpdate                              BEScrollViewScrollUpdate
#define WKBEScrollViewScrollUpdatePhase                         BEScrollViewScrollUpdatePhase
#define WKBEScrollViewScrollUpdatePhaseBegan                    BEScrollViewScrollUpdatePhaseBegan
#define WKBEScrollViewScrollUpdatePhaseChanged                  BEScrollViewScrollUpdatePhaseChanged
#define WKBEScrollViewScrollUpdatePhaseEnded                    BEScrollViewScrollUpdatePhaseEnded
#define WKBEScrollViewScrollUpdatePhaseCancelled                BEScrollViewScrollUpdatePhaseCancelled
// Editing and keyboards
#define WKBETextSuggestion                                      BETextSuggestion
#define WKBETextDocumentContext                                 BETextDocumentContext
#define WKBETextDocumentRequest                                 BETextDocumentRequest
#define WKBETextDocumentRequestOptions                          BETextDocumentRequestOptions
#define WKBETextDocumentRequestOptionNone                       BETextDocumentOptionNone
#define WKBETextDocumentRequestOptionText                       BETextDocumentOptionText
#define WKBETextDocumentRequestOptionAttributedText             BETextDocumentOptionAttributedText
#define WKBETextDocumentRequestOptionTextRects                  BETextDocumentOptionTextRects
#define WKBETextDocumentRequestOptionMarkedTextRects            BETextDocumentOptionMarkedTextRects
#define WKBETextDocumentRequestOptionAutocorrectedRanges        BETextDocumentOptionAutocorrectedRanges
#define WKBEGestureType                                         BEGestureType
#define WKBEGestureTypeLoupe                                    BEGestureTypeLoupe
#define WKBEGestureTypeOneFingerTap                             BEGestureTypeOneFingerTap
#define WKBEGestureTypeDoubleTapAndHold                         BEGestureTypeDoubleTapAndHold
#define WKBEGestureTypeDoubleTap                                BEGestureTypeDoubleTap
#define WKBEGestureTypeOneFingerDoubleTap                       BEGestureTypeOneFingerDoubleTap
#define WKBEGestureTypeOneFingerTripleTap                       BEGestureTypeOneFingerTripleTap
#define WKBEGestureTypeTwoFingerSingleTap                       BEGestureTypeTwoFingerSingleTap
#define WKBEGestureTypeTwoFingerRangedSelectGesture             BEGestureTypeTwoFingerRangedSelectGesture
#define WKBEGestureTypeIMPhraseBoundaryDrag                     BEGestureTypeIMPhraseBoundaryDrag
#define WKBEGestureTypeForceTouch                               BEGestureTypeForceTouch
#define WKBESelectionTouchPhase                                 BESelectionTouchPhase
#define WKBESelectionTouchPhaseStarted                          BESelectionTouchPhaseStarted
#define WKBESelectionTouchPhaseMoved                            BESelectionTouchPhaseMoved
#define WKBESelectionTouchPhaseEnded                            BESelectionTouchPhaseEnded
#define WKBESelectionTouchPhaseEndedMovingForward               BESelectionTouchPhaseEndedMovingForward
#define WKBESelectionTouchPhaseEndedMovingBackward              BESelectionTouchPhaseEndedMovingBackward
#define WKBESelectionTouchPhaseEndedNotMoving                   BESelectionTouchPhaseEndedNotMoving
#define WKBESelectionFlags                                      BESelectionFlags
#define WKBESelectionFlagsNone                                  BESelectionFlagsNone
#define WKBEWordIsNearTap                                       BEWordIsNearTap
#define WKBESelectionFlipped                                    BESelectionFlipped
#define WKBEPhraseBoundaryChanged                               BEPhraseBoundaryChanged
#else
// Scroll view
#define WKBEScrollView                                          UIScrollView
#define WKBEScrollViewDelegate                                  UIScrollViewDelegate
#define WKBEScrollViewScrollUpdate                              UIScrollEvent
#define WKBEScrollViewScrollUpdatePhase                         UIScrollPhase
#define WKBEScrollViewScrollUpdatePhaseBegan                    UIScrollPhaseBegan
#define WKBEScrollViewScrollUpdatePhaseChanged                  UIScrollPhaseChanged
#define WKBEScrollViewScrollUpdatePhaseEnded                    UIScrollPhaseEnded
#define WKBEScrollViewScrollUpdatePhaseCancelled                UIScrollPhaseCancelled
// Editing and keyboards
#define WKBETextSuggestion                                      UITextSuggestion
#define WKBETextDocumentContext                                 UIWKDocumentContext
#define WKBETextDocumentRequest                                 UIWKDocumentRequest
#define WKBETextDocumentRequestOptions                          UIWKDocumentRequestFlags
#define WKBETextDocumentRequestOptionNone                       UIWKDocumentRequestNone
#define WKBETextDocumentRequestOptionText                       UIWKDocumentRequestText
#define WKBETextDocumentRequestOptionAttributedText             UIWKDocumentRequestAttributed
#define WKBETextDocumentRequestOptionTextRects                  UIWKDocumentRequestRects
#define WKBETextDocumentRequestOptionMarkedTextRects            UIWKDocumentRequestMarkedTextRects
#define WKBETextDocumentRequestOptionAutocorrectedRanges        UIWKDocumentRequestAutocorrectedRanges
#define WKBEGestureType                                         UIWKGestureType
#define WKBEGestureTypeLoupe                                    UIWKGestureLoupe
#define WKBEGestureTypeOneFingerTap                             UIWKGestureOneFingerTap
#define WKBEGestureTypeDoubleTapAndHold                         UIWKGestureTapAndAHalf
#define WKBEGestureTypeDoubleTap                                UIWKGestureDoubleTap
#define WKBEGestureTypeOneFingerDoubleTap                       UIWKGestureOneFingerDoubleTap
#define WKBEGestureTypeOneFingerTripleTap                       UIWKGestureOneFingerTripleTap
#define WKBEGestureTypeTwoFingerSingleTap                       UIWKGestureTwoFingerSingleTap
#define WKBEGestureTypeTwoFingerRangedSelectGesture             UIWKGestureTwoFingerRangedSelectGesture
#define WKBEGestureTypeIMPhraseBoundaryDrag                     UIWKGesturePhraseBoundary
#define WKBESelectionTouchPhase                                 UIWKSelectionTouch
#define WKBESelectionTouchPhaseStarted                          UIWKSelectionTouchStarted
#define WKBESelectionTouchPhaseMoved                            UIWKSelectionTouchMoved
#define WKBESelectionTouchPhaseEnded                            UIWKSelectionTouchEnded
#define WKBESelectionTouchPhaseEndedMovingForward               UIWKSelectionTouchEndedMovingForward
#define WKBESelectionTouchPhaseEndedMovingBackward              UIWKSelectionTouchEndedMovingBackward
#define WKBESelectionTouchPhaseEndedNotMoving                   UIWKSelectionTouchEndedNotMoving
#define WKBESelectionFlags                                      UIWKSelectionFlags
#define WKBESelectionFlagsNone                                  UIWKNone
#define WKBEWordIsNearTap                                       UIWKWordIsNearTap
#define WKBESelectionFlipped                                    UIWKSelectionFlipped
#define WKBEPhraseBoundaryChanged                               UIWKPhraseBoundaryChanged
#endif
