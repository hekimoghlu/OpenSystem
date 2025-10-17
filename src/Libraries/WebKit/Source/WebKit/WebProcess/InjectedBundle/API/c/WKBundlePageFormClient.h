/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#ifndef WKBundlePageFormClient_h
#define WKBundlePageFormClient_h

#include <WebKit/WKBase.h>
#include <WebKit/WKBundlePageEditorClient.h>

typedef void (*WKBundlePageTextFieldDidBeginEditingCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlInputElementHandle, WKBundleFrameRef frame, const void* clientInfo);
typedef void (*WKBundlePageTextFieldDidEndEditingCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlInputElementHandle, WKBundleFrameRef frame, const void* clientInfo);
typedef void (*WKBundlePageTextDidChangeInTextFieldCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlInputElementHandle, WKBundleFrameRef frame, const void* clientInfo);
typedef void (*WKBundlePageTextDidChangeInTextAreaCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlTextAreaElementHandle, WKBundleFrameRef frame, const void* clientInfo);
typedef bool (*WKBundlePageShouldPerformActionInTextFieldCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlInputElementHandle, WKInputFieldActionType actionType, WKBundleFrameRef frame, const void* clientInfo);
typedef void (*WKBundlePageWillSubmitFormCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlFormElementHandle, WKBundleFrameRef frame, WKBundleFrameRef sourceFrame, WKDictionaryRef values, WKTypeRef* userData, const void* clientInfo);
typedef void (*WKBundlePageWillSendSubmitEventCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlFormElementHandle, WKBundleFrameRef frame, WKBundleFrameRef sourceFrame, WKDictionaryRef values, const void* clientInfo);
typedef void (*WKBundlePageDidFocusTextFieldCallback)(WKBundlePageRef page, WKBundleNodeHandleRef htmlInputElementHandle, WKBundleFrameRef frame, const void* clientInfo);
typedef bool (*WKBundlePageShouldNotifyOnFormChangesCallback)(WKBundlePageRef page, const void* clientInfo);
typedef void (*WKBundlePageDidAssociateFormControlsCallback)(WKBundlePageRef page, WKArrayRef elementHandles, const void* clientInfo);
typedef void (*WKBundlePageDidAssociateFormControlsForFrameCallback)(WKBundlePageRef page, WKArrayRef elementHandles, WKBundleFrameRef frame, const void* clientInfo);

typedef struct WKBundlePageFormClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageFormClientBase;

typedef struct WKBundlePageFormClientV0 {
    WKBundlePageFormClientBase                                          base;

    // Version 0.
    WKBundlePageTextFieldDidBeginEditingCallback                        textFieldDidBeginEditing;
    WKBundlePageTextFieldDidEndEditingCallback                          textFieldDidEndEditing;
    WKBundlePageTextDidChangeInTextFieldCallback                        textDidChangeInTextField;
    WKBundlePageTextDidChangeInTextAreaCallback                         textDidChangeInTextArea;
    WKBundlePageShouldPerformActionInTextFieldCallback                  shouldPerformActionInTextField;
    WKBundlePageWillSubmitFormCallback                                  willSubmitForm;
} WKBundlePageFormClientV0;

typedef struct WKBundlePageFormClientV1 {
    WKBundlePageFormClientBase                                          base;

    // Version 0.
    WKBundlePageTextFieldDidBeginEditingCallback                        textFieldDidBeginEditing;
    WKBundlePageTextFieldDidEndEditingCallback                          textFieldDidEndEditing;
    WKBundlePageTextDidChangeInTextFieldCallback                        textDidChangeInTextField;
    WKBundlePageTextDidChangeInTextAreaCallback                         textDidChangeInTextArea;
    WKBundlePageShouldPerformActionInTextFieldCallback                  shouldPerformActionInTextField;
    WKBundlePageWillSubmitFormCallback                                  willSubmitForm;

    // Version 1.
    WKBundlePageWillSendSubmitEventCallback                             willSendSubmitEvent;
} WKBundlePageFormClientV1;

typedef struct WKBundlePageFormClientV2 {
    WKBundlePageFormClientBase                                          base;

    // Version 0.
    WKBundlePageTextFieldDidBeginEditingCallback                        textFieldDidBeginEditing;
    WKBundlePageTextFieldDidEndEditingCallback                          textFieldDidEndEditing;
    WKBundlePageTextDidChangeInTextFieldCallback                        textDidChangeInTextField;
    WKBundlePageTextDidChangeInTextAreaCallback                         textDidChangeInTextArea;
    WKBundlePageShouldPerformActionInTextFieldCallback                  shouldPerformActionInTextField;
    WKBundlePageWillSubmitFormCallback                                  willSubmitForm;

    // Version 1.
    WKBundlePageWillSendSubmitEventCallback                             willSendSubmitEvent;

    // version 2.
    WKBundlePageDidFocusTextFieldCallback                               didFocusTextField;
    WKBundlePageShouldNotifyOnFormChangesCallback                       shouldNotifyOnFormChanges;
    WKBundlePageDidAssociateFormControlsCallback                        didAssociateFormControls;
} WKBundlePageFormClientV2;

typedef struct WKBundlePageFormClientV3 {
    WKBundlePageFormClientBase                                          base;

    // Version 0.
    WKBundlePageTextFieldDidBeginEditingCallback                        textFieldDidBeginEditing;
    WKBundlePageTextFieldDidEndEditingCallback                          textFieldDidEndEditing;
    WKBundlePageTextDidChangeInTextFieldCallback                        textDidChangeInTextField;
    WKBundlePageTextDidChangeInTextAreaCallback                         textDidChangeInTextArea;
    WKBundlePageShouldPerformActionInTextFieldCallback                  shouldPerformActionInTextField;
    WKBundlePageWillSubmitFormCallback                                  willSubmitForm;

    // Version 1.
    WKBundlePageWillSendSubmitEventCallback                             willSendSubmitEvent;

    // version 2.
    WKBundlePageDidFocusTextFieldCallback                               didFocusTextField;
    WKBundlePageShouldNotifyOnFormChangesCallback                       shouldNotifyOnFormChanges;
    WKBundlePageDidAssociateFormControlsCallback                        didAssociateFormControls;

    // version 3.
    WKBundlePageDidAssociateFormControlsForFrameCallback                didAssociateFormControlsForFrame;
} WKBundlePageFormClientV3;

#endif // WKBundlePageFormClient_h
