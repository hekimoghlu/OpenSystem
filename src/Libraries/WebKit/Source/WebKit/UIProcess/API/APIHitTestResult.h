/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#include "WebHitTestResultData.h"
#include "WebPageProxy.h"
#include <WebCore/DictionaryPopupInfo.h>
#include <WebCore/FloatPoint.h>
#include <WebCore/IntRect.h>
#include <WebCore/PageOverlay.h>
#include <WebCore/SharedMemory.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebCore {
class HitTestResult;
}

namespace API {

class WebFrame;

class HitTestResult : public API::ObjectImpl<API::Object::Type::HitTestResult> {
public:
    static Ref<HitTestResult> create(const WebKit::WebHitTestResultData&, WebKit::WebPageProxy*);

    WTF::String absoluteImageURL() const { return m_data.absoluteImageURL; }
    WTF::String absolutePDFURL() const { return m_data.absolutePDFURL; }
    WTF::String absoluteLinkURL() const { return m_data.absoluteLinkURL; }
    WTF::String absoluteMediaURL() const { return m_data.absoluteMediaURL; }

    WTF::String linkLabel() const { return m_data.linkLabel; }
    WTF::String linkTitle() const { return m_data.linkTitle; }
    WTF::String linkLocalDataMIMEType() const { return m_data.linkLocalDataMIMEType; }
    WTF::String linkSuggestedFilename() const { return m_data.linkSuggestedFilename; }
    WTF::String imageSuggestedFilename() const { return m_data.imageSuggestedFilename; }
    WTF::String lookupText() const { return m_data.lookupText; }
    WTF::String sourceImageMIMEType() const { return m_data.sourceImageMIMEType; }

    bool isContentEditable() const { return m_data.isContentEditable; }

    WebCore::IntRect elementBoundingBox() const { return m_data.elementBoundingBox; }

    bool isScrollbar() const { return m_data.isScrollbar != WebKit::WebHitTestResultData::IsScrollbar::No; }

    bool isSelected() const { return m_data.isSelected; }

    bool isTextNode() const { return m_data.isTextNode; }

    bool isOverTextInsideFormControlElement() const { return m_data.isOverTextInsideFormControlElement; }

    bool isDownloadableMedia() const { return m_data.isDownloadableMedia; }

    bool mediaIsInFullscreen() const { return m_data.mediaIsInFullscreen; }

    WebKit::WebHitTestResultData::ElementType elementType() const { return m_data.elementType; }

    WebKit::WebPageProxy* page() { return m_page.get(); }

    const std::optional<WebKit::FrameInfoData>& frameInfo() const { return m_data.frameInfo; }

    bool hasLocalDataForLinkURL() const { return m_data.hasLocalDataForLinkURL; }

private:
    explicit HitTestResult(const WebKit::WebHitTestResultData& hitTestResultData, WebKit::WebPageProxy* page)
        : m_data(hitTestResultData)
        , m_page(page)
    {
    }

    WebKit::WebHitTestResultData m_data;
    WeakPtr<WebKit::WebPageProxy> m_page;
};

} // namespace API
