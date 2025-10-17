/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

#include "APIObject.h"
#include "ImageOptions.h"
#include <JavaScriptCore/JSBase.h>
#include <WebCore/ActiveDOMObject.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class IntRect;
class Node;
enum class AutoFillButtonType : uint8_t;
}

namespace WebKit {

class InjectedBundleRangeHandle;
class InjectedBundleScriptWorld;
class WebFrame;
class WebImage;

class InjectedBundleNodeHandle : public API::ObjectImpl<API::Object::Type::BundleNodeHandle>, public WebCore::ActiveDOMObject, public CanMakeWeakPtr<InjectedBundleNodeHandle> {
public:
    static RefPtr<InjectedBundleNodeHandle> getOrCreate(JSContextRef, JSObjectRef);
    static RefPtr<InjectedBundleNodeHandle> getOrCreate(WebCore::Node*);
    static Ref<InjectedBundleNodeHandle> getOrCreate(WebCore::Node&);

    virtual ~InjectedBundleNodeHandle();

    WebCore::Node* coreNode();

    // Convenience DOM Operations
    RefPtr<InjectedBundleNodeHandle> document();

    // Additional DOM Operations
    // Note: These should only be operations that are not exposed to JavaScript.
    WebCore::IntRect elementBounds();
    WebCore::IntRect absoluteBoundingRect(bool*);
    RefPtr<WebImage> renderedImage(SnapshotOptions, bool shouldExcludeOverflow, const std::optional<float>& bitmapWidth = std::nullopt);
    RefPtr<InjectedBundleRangeHandle> visibleRange();
    void setHTMLInputElementValueForUser(const String&);
    void setHTMLInputElementSpellcheckEnabled(bool);
    bool isHTMLInputElementAutoFilled() const;
    bool isHTMLInputElementAutoFilledAndViewable() const;
    bool isHTMLInputElementAutoFilledAndObscured() const;
    void setHTMLInputElementAutoFilled(bool);
    void setHTMLInputElementAutoFilledAndViewable(bool);
    void setHTMLInputElementAutoFilledAndObscured(bool);
    bool isHTMLInputElementAutoFillButtonEnabled() const;
    void setHTMLInputElementAutoFillButtonEnabled(WebCore::AutoFillButtonType);
    WebCore::AutoFillButtonType htmlInputElementAutoFillButtonType() const;
    WebCore::AutoFillButtonType htmlInputElementLastAutoFillButtonType() const;
    bool isAutoFillAvailable() const;
    void setAutoFillAvailable(bool);
    WebCore::IntRect htmlInputElementAutoFillButtonBounds();
    bool htmlInputElementLastChangeWasUserEdit();
    bool htmlTextAreaElementLastChangeWasUserEdit();
    bool isTextField() const;
    bool isSelectElement() const;
    bool isSelectableTextNode() const;
    
    RefPtr<InjectedBundleNodeHandle> htmlTableCellElementCellAbove();

    RefPtr<WebFrame> documentFrame();
    RefPtr<WebFrame> htmlIFrameElementContentFrame();

    // ActiveDOMObject.
    void ref() const final { API::ObjectImpl<API::Object::Type::BundleNodeHandle>::ref(); }
    void deref() const final { API::ObjectImpl<API::Object::Type::BundleNodeHandle>::deref(); }

private:
    static Ref<InjectedBundleNodeHandle> create(WebCore::Node&);
    InjectedBundleNodeHandle(WebCore::Node&);

    // ActiveDOMObject.
    void stop() final;

    RefPtr<WebCore::Node> m_node;
};

} // namespace WebKit
