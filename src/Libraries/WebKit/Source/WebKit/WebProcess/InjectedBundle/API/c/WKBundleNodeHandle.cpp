/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "config.h"
#include "WKBundleNodeHandle.h"
#include "WKBundleNodeHandlePrivate.h"

#include "InjectedBundleNodeHandle.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WebFrame.h"
#include "WebImage.h"
#include <WebCore/HTMLTextFormControlElement.h>

static WebCore::AutoFillButtonType toAutoFillButtonType(WKAutoFillButtonType wkAutoFillButtonType)
{
    switch (wkAutoFillButtonType) {
    case kWKAutoFillButtonTypeNone:
        return WebCore::AutoFillButtonType::None;
    case kWKAutoFillButtonTypeContacts:
        return WebCore::AutoFillButtonType::Contacts;
    case kWKAutoFillButtonTypeCredentials:
        return WebCore::AutoFillButtonType::Credentials;
    case kWKAutoFillButtonTypeStrongPassword:
        return WebCore::AutoFillButtonType::StrongPassword;
    case kWKAutoFillButtonTypeCreditCard:
        return WebCore::AutoFillButtonType::CreditCard;
    case kWKAutoFillButtonTypeLoading:
        return WebCore::AutoFillButtonType::Loading;
    }
    ASSERT_NOT_REACHED();
    return WebCore::AutoFillButtonType::None;
}

static WKAutoFillButtonType toWKAutoFillButtonType(WebCore::AutoFillButtonType autoFillButtonType)
{
    switch (autoFillButtonType) {
    case WebCore::AutoFillButtonType::None:
        return kWKAutoFillButtonTypeNone;
    case WebCore::AutoFillButtonType::Contacts:
        return kWKAutoFillButtonTypeContacts;
    case WebCore::AutoFillButtonType::Credentials:
        return kWKAutoFillButtonTypeCredentials;
    case WebCore::AutoFillButtonType::StrongPassword:
        return kWKAutoFillButtonTypeStrongPassword;
    case WebCore::AutoFillButtonType::CreditCard:
        return kWKAutoFillButtonTypeCreditCard;
    case WebCore::AutoFillButtonType::Loading:
        return kWKAutoFillButtonTypeLoading;
    }
    ASSERT_NOT_REACHED();
    return kWKAutoFillButtonTypeNone;
}

WKTypeID WKBundleNodeHandleGetTypeID()
{
    return WebKit::toAPI(WebKit::InjectedBundleNodeHandle::APIType);
}

WKBundleNodeHandleRef WKBundleNodeHandleCreate(JSContextRef contextRef, JSObjectRef objectRef)
{
    RefPtr<WebKit::InjectedBundleNodeHandle> nodeHandle = WebKit::InjectedBundleNodeHandle::getOrCreate(contextRef, objectRef);
    return toAPI(nodeHandle.leakRef());
}

WKBundleNodeHandleRef WKBundleNodeHandleCopyDocument(WKBundleNodeHandleRef nodeHandleRef)
{
    RefPtr<WebKit::InjectedBundleNodeHandle> nodeHandle = WebKit::toImpl(nodeHandleRef)->document();
    return toAPI(nodeHandle.leakRef());
}

WKRect WKBundleNodeHandleGetRenderRect(WKBundleNodeHandleRef nodeHandleRef, bool* isReplaced)
{
    ASSERT_NOT_REACHED();
    return { };
}

WKImageRef WKBundleNodeHandleCopySnapshotWithOptions(WKBundleNodeHandleRef nodeHandleRef, WKSnapshotOptions options)
{
    RefPtr<WebKit::WebImage> image = WebKit::toImpl(nodeHandleRef)->renderedImage(WebKit::toSnapshotOptions(options), options & kWKSnapshotOptionsExcludeOverflow);
    return toAPI(image.leakRef());
}

WKBundleRangeHandleRef WKBundleNodeHandleCopyVisibleRange(WKBundleNodeHandleRef)
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

WKRect WKBundleNodeHandleGetElementBounds(WKBundleNodeHandleRef elementHandleRef)
{
    return WebKit::toAPI(WebKit::toImpl(elementHandleRef)->elementBounds());
}

void WKBundleNodeHandleSetHTMLInputElementValueForUser(WKBundleNodeHandleRef htmlInputElementHandleRef, WKStringRef valueRef)
{
    WebKit::toImpl(htmlInputElementHandleRef)->setHTMLInputElementValueForUser(WebKit::toWTFString(valueRef));
}

void WKBundleNodeHandleSetHTMLInputElementSpellcheckEnabled(WKBundleNodeHandleRef htmlInputElementHandleRef, bool enabled)
{
    WebKit::toImpl(htmlInputElementHandleRef)->setHTMLInputElementSpellcheckEnabled(enabled);
}

bool WKBundleNodeHandleGetHTMLInputElementAutoFilled(WKBundleNodeHandleRef)
{
    ASSERT_NOT_REACHED();
    return false;
}

void WKBundleNodeHandleSetHTMLInputElementAutoFilled(WKBundleNodeHandleRef htmlInputElementHandleRef, bool filled)
{
    WebKit::toImpl(htmlInputElementHandleRef)->setHTMLInputElementAutoFilled(filled);
}

void WKBundleNodeHandleSetHTMLInputElementAutoFilledAndViewable(WKBundleNodeHandleRef htmlInputElementHandleRef, bool autoFilledAndViewable)
{
    WebKit::toImpl(htmlInputElementHandleRef)->setHTMLInputElementAutoFilledAndViewable(autoFilledAndViewable);
}

void WKBundleNodeHandleSetHTMLInputElementAutoFilledAndObscured(WKBundleNodeHandleRef htmlInputElementHandleRef, bool autoFilledAndObscured)
{
    WebKit::toImpl(htmlInputElementHandleRef)->setHTMLInputElementAutoFilledAndObscured(autoFilledAndObscured);
}

bool WKBundleNodeHandleGetHTMLInputElementAutoFillButtonEnabled(WKBundleNodeHandleRef)
{
    ASSERT_NOT_REACHED();
    return false;
}

void WKBundleNodeHandleSetHTMLInputElementAutoFillButtonEnabledWithButtonType(WKBundleNodeHandleRef htmlInputElementHandleRef, WKAutoFillButtonType autoFillButtonType)
{
    WebKit::toImpl(htmlInputElementHandleRef)->setHTMLInputElementAutoFillButtonEnabled(toAutoFillButtonType(autoFillButtonType));
}

WKAutoFillButtonType WKBundleNodeHandleGetHTMLInputElementAutoFillButtonType(WKBundleNodeHandleRef htmlInputElementHandleRef)
{
    return toWKAutoFillButtonType(WebKit::toImpl(htmlInputElementHandleRef)->htmlInputElementAutoFillButtonType());
}

WKAutoFillButtonType WKBundleNodeHandleGetHTMLInputElementLastAutoFillButtonType(WKBundleNodeHandleRef htmlInputElementHandleRef)
{
    return toWKAutoFillButtonType(WebKit::toImpl(htmlInputElementHandleRef)->htmlInputElementLastAutoFillButtonType());
}

bool WKBundleNodeHandleGetHTMLInputElementAutoFillAvailable(WKBundleNodeHandleRef)
{
    ASSERT_NOT_REACHED();
    return false;
}

void WKBundleNodeHandleSetHTMLInputElementAutoFillAvailable(WKBundleNodeHandleRef htmlInputElementHandleRef, bool autoFillAvailable)
{
    WebKit::toImpl(htmlInputElementHandleRef)->setAutoFillAvailable(autoFillAvailable);
}

WKRect WKBundleNodeHandleGetHTMLInputElementAutoFillButtonBounds(WKBundleNodeHandleRef)
{
    ASSERT_NOT_REACHED();
    return { };
}

bool WKBundleNodeHandleGetHTMLInputElementLastChangeWasUserEdit(WKBundleNodeHandleRef htmlInputElementHandleRef)
{
    return WebKit::toImpl(htmlInputElementHandleRef)->htmlInputElementLastChangeWasUserEdit();
}

bool WKBundleNodeHandleGetHTMLTextAreaElementLastChangeWasUserEdit(WKBundleNodeHandleRef htmlTextAreaElementHandleRef)
{
    return WebKit::toImpl(htmlTextAreaElementHandleRef)->htmlTextAreaElementLastChangeWasUserEdit();
}

WKBundleNodeHandleRef WKBundleNodeHandleCopyHTMLTableCellElementCellAbove(WKBundleNodeHandleRef)
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

WKBundleFrameRef WKBundleNodeHandleCopyDocumentFrame(WKBundleNodeHandleRef documentHandleRef)
{
    RefPtr<WebKit::WebFrame> frame = WebKit::toImpl(documentHandleRef)->documentFrame();
    return toAPI(frame.leakRef());
}

WKBundleFrameRef WKBundleNodeHandleCopyHTMLFrameElementContentFrame(WKBundleNodeHandleRef htmlFrameElementHandleRef)
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

WKBundleFrameRef WKBundleNodeHandleCopyHTMLIFrameElementContentFrame(WKBundleNodeHandleRef htmlIFrameElementHandleRef)
{
    RefPtr<WebKit::WebFrame> frame = WebKit::toImpl(htmlIFrameElementHandleRef)->htmlIFrameElementContentFrame();
    return toAPI(frame.leakRef());
}

bool WKBundleNodeHandleGetHTMLInputElementAutofilled(WKBundleNodeHandleRef htmlInputElementHandleRef)
{
    ASSERT_NOT_REACHED();
    return false;
}

void WKBundleNodeHandleSetHTMLInputElementAutofilled(WKBundleNodeHandleRef handle, bool enabled)
{
    WKBundleNodeHandleSetHTMLInputElementAutoFilled(handle, enabled);
}

void WKBundleNodeHandleSetHTMLInputElementAutoFillButtonEnabled(WKBundleNodeHandleRef, bool)
{
    // FIXME: Would put ASSERT_NOT_REACHED() here but some compilers are warning the function is "noreturn".
}

WKBundleFrameRef WKBundleNodeHandleCopyOwningDocumentFrame(WKBundleNodeHandleRef documentHandleRef)
{
    if (RefPtr document = WebKit::toImpl(documentHandleRef)->document())
        return toAPI(document->documentFrame().leakRef());
    return nullptr;
}
