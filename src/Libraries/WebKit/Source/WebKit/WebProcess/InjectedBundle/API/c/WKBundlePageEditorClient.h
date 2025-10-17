/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
#ifndef WKBundlePageEditorClient_h
#define WKBundlePageEditorClient_h

#include <WebKit/WKBase.h>

enum {
    kWKInsertActionTyped = 0,
    kWKInsertActionPasted = 1,
    kWKInsertActionDropped = 2
};
typedef uint32_t WKInsertActionType;

enum {
    kWKAffinityUpstream,
    kWKAffinityDownstream
};
typedef uint32_t WKAffinityType;

enum {
    WKInputFieldActionTypeMoveUp,
    WKInputFieldActionTypeMoveDown,
    WKInputFieldActionTypeCancel,
    WKInputFieldActionTypeInsertTab,
    WKInputFieldActionTypeInsertBacktab,
    WKInputFieldActionTypeInsertNewline,
    WKInputFieldActionTypeInsertDelete
};
typedef uint32_t WKInputFieldActionType;

typedef bool (*WKBundlePageShouldBeginEditingCallback)(WKBundlePageRef page, WKBundleRangeHandleRef range, const void* clientInfo);
typedef bool (*WKBundlePageShouldEndEditingCallback)(WKBundlePageRef page, WKBundleRangeHandleRef range, const void* clientInfo);
typedef bool (*WKBundlePageShouldInsertNodeCallback)(WKBundlePageRef page, WKBundleNodeHandleRef node, WKBundleRangeHandleRef rangeToReplace, WKInsertActionType action, const void* clientInfo);
typedef bool (*WKBundlePageShouldInsertTextCallback)(WKBundlePageRef page, WKStringRef string, WKBundleRangeHandleRef rangeToReplace, WKInsertActionType action, const void* clientInfo);
typedef bool (*WKBundlePageShouldDeleteRangeCallback)(WKBundlePageRef page, WKBundleRangeHandleRef range, const void* clientInfo);
typedef bool (*WKBundlePageShouldChangeSelectedRange)(WKBundlePageRef page, WKBundleRangeHandleRef fromRange, WKBundleRangeHandleRef toRange, WKAffinityType affinity, bool stillSelecting, const void* clientInfo);
typedef bool (*WKBundlePageShouldApplyStyle)(WKBundlePageRef page, WKBundleCSSStyleDeclarationRef style, WKBundleRangeHandleRef range, const void* clientInfo);
typedef void (*WKBundlePageEditingNotification)(WKBundlePageRef page, WKStringRef notificationName, const void* clientInfo);
typedef void (*WKBundlePageWillWriteToPasteboard)(WKBundlePageRef page, WKBundleRangeHandleRef range,  const void* clientInfo);
typedef void (*WKBundlePageGetPasteboardDataForRange)(WKBundlePageRef page, WKBundleRangeHandleRef range, WKArrayRef* pasteboardTypes, WKArrayRef* pasteboardData, const void* clientInfo);
typedef WKStringRef (*WKBundlePageReplacementURLForResource)(WKBundlePageRef, WKDataRef resourceData, WKStringRef mimeType, const void* clientInfo);
typedef void (*WKBundlePageDidWriteToPasteboard)(WKBundlePageRef page, const void* clientInfo);
typedef bool (*WKBundlePagePerformTwoStepDrop)(WKBundlePageRef page, WKBundleNodeHandleRef fragment, WKBundleRangeHandleRef destination, bool isMove, const void* clientInfo);

typedef struct WKBundlePageEditorClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageEditorClientBase;

typedef struct WKBundlePageEditorClientV0 {
    WKBundlePageEditorClientBase                                        base;

    // Version 0.
    WKBundlePageShouldBeginEditingCallback                              shouldBeginEditing;
    WKBundlePageShouldEndEditingCallback                                shouldEndEditing;
    WKBundlePageShouldInsertNodeCallback                                shouldInsertNode;
    WKBundlePageShouldInsertTextCallback                                shouldInsertText;
    WKBundlePageShouldDeleteRangeCallback                               shouldDeleteRange;
    WKBundlePageShouldChangeSelectedRange                               shouldChangeSelectedRange;
    WKBundlePageShouldApplyStyle                                        shouldApplyStyle;
    WKBundlePageEditingNotification                                     didBeginEditing;
    WKBundlePageEditingNotification                                     didEndEditing;
    WKBundlePageEditingNotification                                     didChange;
    WKBundlePageEditingNotification                                     didChangeSelection;
} WKBundlePageEditorClientV0;

typedef struct WKBundlePageEditorClientV1 {
    WKBundlePageEditorClientBase                                        base;

    // Version 0.
    WKBundlePageShouldBeginEditingCallback                              shouldBeginEditing;
    WKBundlePageShouldEndEditingCallback                                shouldEndEditing;
    WKBundlePageShouldInsertNodeCallback                                shouldInsertNode;
    WKBundlePageShouldInsertTextCallback                                shouldInsertText;
    WKBundlePageShouldDeleteRangeCallback                               shouldDeleteRange;
    WKBundlePageShouldChangeSelectedRange                               shouldChangeSelectedRange;
    WKBundlePageShouldApplyStyle                                        shouldApplyStyle;
    WKBundlePageEditingNotification                                     didBeginEditing;
    WKBundlePageEditingNotification                                     didEndEditing;
    WKBundlePageEditingNotification                                     didChange;
    WKBundlePageEditingNotification                                     didChangeSelection;

    // Version 1.
    WKBundlePageWillWriteToPasteboard                                   willWriteToPasteboard;
    WKBundlePageGetPasteboardDataForRange                               getPasteboardDataForRange;
    WKBundlePageDidWriteToPasteboard                                    didWriteToPasteboard;
    WKBundlePagePerformTwoStepDrop                                      performTwoStepDrop;
} WKBundlePageEditorClientV1;

typedef struct WKBundlePageEditorClientV2 {
    WKBundlePageEditorClientBase                                        base;

    // Version 0.
    WKBundlePageShouldBeginEditingCallback                              shouldBeginEditing;
    WKBundlePageShouldEndEditingCallback                                shouldEndEditing;
    WKBundlePageShouldInsertNodeCallback                                shouldInsertNode;
    WKBundlePageShouldInsertTextCallback                                shouldInsertText;
    WKBundlePageShouldDeleteRangeCallback                               shouldDeleteRange;
    WKBundlePageShouldChangeSelectedRange                               shouldChangeSelectedRange;
    WKBundlePageShouldApplyStyle                                        shouldApplyStyle;
    WKBundlePageEditingNotification                                     didBeginEditing;
    WKBundlePageEditingNotification                                     didEndEditing;
    WKBundlePageEditingNotification                                     didChange;
    WKBundlePageEditingNotification                                     didChangeSelection;

    // Version 1.
    WKBundlePageWillWriteToPasteboard                                   willWriteToPasteboard;
    WKBundlePageGetPasteboardDataForRange                               getPasteboardDataForRange;
    WKBundlePageDidWriteToPasteboard                                    didWriteToPasteboard;
    WKBundlePagePerformTwoStepDrop                                      performTwoStepDrop;

    // Version 2.
    WKBundlePageReplacementURLForResource                               replacementURLForResource;
} WKBundlePageEditorClientV2;

#endif // WKBundlePageEditorClient_h
