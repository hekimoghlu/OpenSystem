/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#import "WKWebProcessPlugInNodeHandleInternal.h"

#import "CocoaImage.h"
#import "WKSharedAPICast.h"
#import "WKWebProcessPlugInFrameInternal.h"
#import "WebImage.h"
#import <WebCore/HTMLTextFormControlElement.h>
#import <WebCore/IntRect.h>
#import <WebCore/NativeImage.h>
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKWebProcessPlugInNodeHandle {
    API::ObjectStorage<WebKit::InjectedBundleNodeHandle> _nodeHandle;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWebProcessPlugInNodeHandle.class, self))
        return;
    _nodeHandle->~InjectedBundleNodeHandle();
    [super dealloc];
}

+ (WKWebProcessPlugInNodeHandle *)nodeHandleWithJSValue:(JSValue *)value inContext:(JSContext *)context
{
    JSContextRef contextRef = [context JSGlobalContextRef];
    JSObjectRef objectRef = JSValueToObject(contextRef, [value JSValueRef], nullptr);
    return WebKit::wrapper(WebKit::InjectedBundleNodeHandle::getOrCreate(contextRef, objectRef)).autorelease();
}

- (WKWebProcessPlugInFrame *)htmlIFrameElementContentFrame
{
    return WebKit::wrapper(_nodeHandle->htmlIFrameElementContentFrame()).autorelease();
}

- (CocoaImage *)renderedImageWithOptions:(WKSnapshotOptions)options
{
    return [self renderedImageWithOptions:options width:nil];
}

- (CocoaImage *)renderedImageWithOptions:(WKSnapshotOptions)options width:(NSNumber *)width
{
    std::optional<float> optionalWidth;
    if (width)
        optionalWidth = width.floatValue;

    auto image = _nodeHandle->renderedImage(WebKit::toSnapshotOptions(options), options & kWKSnapshotOptionsExcludeOverflow, optionalWidth);
    if (!image)
        return nil;

    auto nativeImage = image->copyNativeImage(WebCore::DontCopyBackingStore);
    if (!nativeImage)
        return nil;

#if USE(APPKIT)
    return adoptNS([[NSImage alloc] initWithCGImage:nativeImage->platformImage().get() size:NSZeroSize]).autorelease();
#else
    return adoptNS([[UIImage alloc] initWithCGImage:nativeImage->platformImage().get()]).autorelease();
#endif
}

- (CGRect)elementBounds
{
    return _nodeHandle->elementBounds();
}

- (BOOL)HTMLInputElementIsAutoFilled
{
    return _nodeHandle->isHTMLInputElementAutoFilled();
}

- (BOOL)HTMLInputElementIsAutoFilledAndViewable
{
    return _nodeHandle->isHTMLInputElementAutoFilledAndViewable();
}

- (BOOL)HTMLInputElementIsAutoFilledAndObscured
{
    return _nodeHandle->isHTMLInputElementAutoFilledAndObscured();
}

- (void)setHTMLInputElementIsAutoFilled:(BOOL)isAutoFilled
{
    _nodeHandle->setHTMLInputElementAutoFilled(isAutoFilled);
}

- (void)setHTMLInputElementIsAutoFilledAndViewable:(BOOL)isAutoFilledAndViewable
{
    _nodeHandle->setHTMLInputElementAutoFilledAndViewable(isAutoFilledAndViewable);
}

- (void)setHTMLInputElementIsAutoFilledAndObscured:(BOOL)isAutoFilledAndObscured
{
    _nodeHandle->setHTMLInputElementAutoFilledAndObscured(isAutoFilledAndObscured);
}

- (BOOL)isHTMLInputElementAutoFillButtonEnabled
{
    return _nodeHandle->isHTMLInputElementAutoFillButtonEnabled();
}

static WebCore::AutoFillButtonType toAutoFillButtonType(_WKAutoFillButtonType autoFillButtonType)
{
    switch (autoFillButtonType) {
    case _WKAutoFillButtonTypeNone:
        return WebCore::AutoFillButtonType::None;
    case _WKAutoFillButtonTypeContacts:
        return WebCore::AutoFillButtonType::Contacts;
    case _WKAutoFillButtonTypeCredentials:
        return WebCore::AutoFillButtonType::Credentials;
    case _WKAutoFillButtonTypeStrongPassword:
        return WebCore::AutoFillButtonType::StrongPassword;
    case _WKAutoFillButtonTypeCreditCard:
        return WebCore::AutoFillButtonType::CreditCard;
    case _WKAutoFillButtonTypeLoading:
        return WebCore::AutoFillButtonType::Loading;
    }
    ASSERT_NOT_REACHED();
    return WebCore::AutoFillButtonType::None;
}

static _WKAutoFillButtonType toWKAutoFillButtonType(WebCore::AutoFillButtonType autoFillButtonType)
{
    switch (autoFillButtonType) {
    case WebCore::AutoFillButtonType::None:
        return _WKAutoFillButtonTypeNone;
    case WebCore::AutoFillButtonType::Contacts:
        return _WKAutoFillButtonTypeContacts;
    case WebCore::AutoFillButtonType::Credentials:
        return _WKAutoFillButtonTypeCredentials;
    case WebCore::AutoFillButtonType::StrongPassword:
        return _WKAutoFillButtonTypeStrongPassword;
    case WebCore::AutoFillButtonType::CreditCard:
        return _WKAutoFillButtonTypeCreditCard;
    case WebCore::AutoFillButtonType::Loading:
        return _WKAutoFillButtonTypeLoading;
    }
    ASSERT_NOT_REACHED();
    return _WKAutoFillButtonTypeNone;

}

- (void)setHTMLInputElementAutoFillButtonEnabledWithButtonType:(_WKAutoFillButtonType)autoFillButtonType
{
    _nodeHandle->setHTMLInputElementAutoFillButtonEnabled(toAutoFillButtonType(autoFillButtonType));
}

- (_WKAutoFillButtonType)htmlInputElementAutoFillButtonType
{
    return toWKAutoFillButtonType(_nodeHandle->htmlInputElementAutoFillButtonType());
}

- (_WKAutoFillButtonType)htmlInputElementLastAutoFillButtonType
{
    return toWKAutoFillButtonType(_nodeHandle->htmlInputElementLastAutoFillButtonType());
}

- (BOOL)HTMLInputElementIsUserEdited
{
    return _nodeHandle->htmlInputElementLastChangeWasUserEdit();
}

- (BOOL)HTMLTextAreaElementIsUserEdited
{
    return _nodeHandle->htmlTextAreaElementLastChangeWasUserEdit();
}

- (BOOL)isSelectElement
{
    return _nodeHandle->isSelectElement();
}

- (BOOL)isSelectableTextNode
{
    return _nodeHandle->isSelectableTextNode();
}

- (BOOL)isTextField
{
    return _nodeHandle->isTextField();
}

- (WKWebProcessPlugInNodeHandle *)HTMLTableCellElementCellAbove
{
    return WebKit::wrapper(_nodeHandle->htmlTableCellElementCellAbove()).autorelease();
}

- (WKWebProcessPlugInFrame *)frame
{
    return WebKit::wrapper(_nodeHandle->document()->documentFrame()).autorelease();
}

- (WebKit::InjectedBundleNodeHandle&)_nodeHandle
{
    return *_nodeHandle;
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_nodeHandle;
}

@end
