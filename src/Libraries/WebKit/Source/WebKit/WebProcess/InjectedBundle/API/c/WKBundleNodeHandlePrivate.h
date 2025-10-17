/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#ifndef WKBundleNodeHandlePrivate_h
#define WKBundleNodeHandlePrivate_h

#include <JavaScriptCore/JavaScript.h>
#include <WebKit/WKBase.h>
#include <WebKit/WKGeometry.h>
#include <WebKit/WKImage.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKAutoFillButtonTypeNone,
    kWKAutoFillButtonTypeCredentials,
    kWKAutoFillButtonTypeContacts,
    kWKAutoFillButtonTypeStrongPassword,
    kWKAutoFillButtonTypeCreditCard,
    kWKAutoFillButtonTypeLoading,
};
typedef uint8_t WKAutoFillButtonType;

WK_EXPORT WKBundleNodeHandleRef WKBundleNodeHandleCreate(JSContextRef context, JSObjectRef object);

/* Convenience Operations */
WK_EXPORT WKBundleNodeHandleRef WKBundleNodeHandleCopyDocument(WKBundleNodeHandleRef nodeHandle);


/* Additional DOM Operations */

WK_EXPORT WKRect WKBundleNodeHandleGetRenderRect(WKBundleNodeHandleRef nodeHandle, bool* isReplaced);
WK_EXPORT WKImageRef WKBundleNodeHandleCopySnapshotWithOptions(WKBundleNodeHandleRef nodeHandle, WKSnapshotOptions options);
WK_EXPORT WKBundleRangeHandleRef WKBundleNodeHandleCopyVisibleRange(WKBundleNodeHandleRef nodeHandle);

/* Element Specific Operations */
WK_EXPORT WKRect WKBundleNodeHandleGetElementBounds(WKBundleNodeHandleRef elementHandle);

/* HTMLInputElement Specific Operations */
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementValueForUser(WKBundleNodeHandleRef htmlInputElementHandle, WKStringRef value);
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementSpellcheckEnabled(WKBundleNodeHandleRef htmlInputElementHandle, bool enabled);
WK_EXPORT bool WKBundleNodeHandleGetHTMLInputElementAutoFilled(WKBundleNodeHandleRef htmlInputElementHandle);
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementAutoFilled(WKBundleNodeHandleRef htmlInputElementHandle, bool filled);
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementAutoFilledAndViewable(WKBundleNodeHandleRef htmlInputElementHandle, bool autoFilledAndViewable);
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementAutoFilledAndObscured(WKBundleNodeHandleRef htmlInputElementHandle, bool autoFilledAndObscured);
WK_EXPORT bool WKBundleNodeHandleGetHTMLInputElementAutoFillButtonEnabled(WKBundleNodeHandleRef htmlInputElementHandle);
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementAutoFillButtonEnabledWithButtonType(WKBundleNodeHandleRef htmlInputElementHandle, WKAutoFillButtonType autoFillButtonType);
WK_EXPORT WKAutoFillButtonType WKBundleNodeHandleGetHTMLInputElementAutoFillButtonType(WKBundleNodeHandleRef htmlInputElementHandle);
WK_EXPORT WKAutoFillButtonType WKBundleNodeHandleGetHTMLInputElementLastAutoFillButtonType(WKBundleNodeHandleRef htmlInputElementHandle);
WK_EXPORT bool WKBundleNodeHandleGetHTMLInputElementAutoFillAvailable(WKBundleNodeHandleRef htmlInputElementHandle);
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementAutoFillAvailable(WKBundleNodeHandleRef htmlInputElementHandle, bool autoFillAvailable);
WK_EXPORT WKRect WKBundleNodeHandleGetHTMLInputElementAutoFillButtonBounds(WKBundleNodeHandleRef htmlInputElementHandle);
WK_EXPORT bool WKBundleNodeHandleGetHTMLInputElementLastChangeWasUserEdit(WKBundleNodeHandleRef htmlInputElementHandle);

/* HTMLTextAreaElement Specific Operations */
WK_EXPORT bool WKBundleNodeHandleGetHTMLTextAreaElementLastChangeWasUserEdit(WKBundleNodeHandleRef htmlTextAreaElementHandle);
    
/* HTMLTableCellElement Specific Operations */
WK_EXPORT WKBundleNodeHandleRef WKBundleNodeHandleCopyHTMLTableCellElementCellAbove(WKBundleNodeHandleRef htmlTableCellElementHandle);

/* Document Specific Operations */
WK_EXPORT WKBundleFrameRef WKBundleNodeHandleCopyDocumentFrame(WKBundleNodeHandleRef documentHandle);

/* HTMLFrameElement Specific Operations */
WK_EXPORT WKBundleFrameRef WKBundleNodeHandleCopyHTMLFrameElementContentFrame(WKBundleNodeHandleRef htmlFrameElementHandle);

/* HTMLIFrameElement Specific Operations */
WK_EXPORT WKBundleFrameRef WKBundleNodeHandleCopyHTMLIFrameElementContentFrame(WKBundleNodeHandleRef htmlIFrameElementHandle);


/* Deprecated - use WKBundleNodeHandleGetHTMLInputElementAutoFilled(WKBundleNodeHandleRef) */
WK_EXPORT bool WKBundleNodeHandleGetHTMLInputElementAutofilled(WKBundleNodeHandleRef htmlInputElementHandle);
/* Deprecated - use WKBundleNodeHandleSetHTMLInputElementAutoFilled(WKBundleNodeHandleRef, bool) */
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementAutofilled(WKBundleNodeHandleRef htmlInputElementHandle, bool filled);
/* Deprecated - use WKBundleNodeHandleSetHTMLInputElementAutoFillButtonEnabled(WKBundleNodeHandleRef, WKAutoFillButtonType)*/
WK_EXPORT void WKBundleNodeHandleSetHTMLInputElementAutoFillButtonEnabled(WKBundleNodeHandleRef htmlInputElementHandle, bool enabled);

WK_EXPORT WKBundleFrameRef WKBundleNodeHandleCopyOwningDocumentFrame(WKBundleNodeHandleRef);

#ifdef __cplusplus
}
#endif

#endif /* WKBundleNodeHandlePrivate_h */
