/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include "FrameInfoData.h"
#include <WebCore/DictionaryPopupInfo.h>
#include <WebCore/FloatPoint.h>
#include <WebCore/IntRect.h>
#include <WebCore/PageOverlay.h>
#include <WebCore/RemoteUserInputEventData.h>
#include <WebCore/ShareableBitmap.h>
#include <WebCore/SharedMemory.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

#if HAVE(SECURE_ACTION_CONTEXT)
OBJC_CLASS DDSecureActionContext;
using WKDDActionContext = DDSecureActionContext;
#else
OBJC_CLASS DDActionContext;
using WKDDActionContext = DDActionContext;
#endif

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebCore {
class HitTestResult;
class LocalFrame;
class NavigationAction;
}

namespace WebKit {

#if PLATFORM(MAC)
struct WebHitTestResultPlatformData {
    struct DetectedDataActionContext {
        RetainPtr<WKDDActionContext> context;
        struct MarkableTraits {
            static bool isEmptyValue(const DetectedDataActionContext& context) { return !context.context; }
            static DetectedDataActionContext emptyValue() { return { nullptr }; }
        };
    };
    Markable<DetectedDataActionContext> detectedDataActionContext;
    WebCore::FloatRect detectedDataBoundingBox;
    RefPtr<WebCore::TextIndicator> detectedDataTextIndicator;
    WebCore::PageOverlay::PageOverlayID detectedDataOriginatingPageOverlay;
};
#endif

struct WebHitTestResultData {
    String absoluteImageURL;
    String absolutePDFURL;
    String absoluteLinkURL;
    String absoluteMediaURL;
    String linkLabel;
    String linkTitle;
    String linkSuggestedFilename;
    String imageSuggestedFilename;
    bool isContentEditable;
    WebCore::IntRect elementBoundingBox;
    enum class IsScrollbar : uint8_t { No, Vertical, Horizontal };
    IsScrollbar isScrollbar;
    bool isSelected;
    bool isTextNode;
    bool isOverTextInsideFormControlElement;
    bool isDownloadableMedia;
    bool mediaIsInFullscreen;
    bool isActivePDFAnnotation;
    enum class ElementType : uint8_t { None, Audio, Video };
    ElementType elementType;
    std::optional<FrameInfoData> frameInfo;
    std::optional<WebCore::RemoteUserInputEventData> remoteUserInputEventData;

    String lookupText;
    String toolTipText;
    String imageText;
    RefPtr<WebCore::SharedMemory> imageSharedMemory;
    RefPtr<WebCore::ShareableBitmap> imageBitmap;
    String sourceImageMIMEType;
    String linkLocalDataMIMEType;
    bool hasLocalDataForLinkURL;
    bool hasEntireImage;

#if PLATFORM(MAC)
    WebHitTestResultPlatformData platformData;
#endif
    
    WebCore::DictionaryPopupInfo dictionaryPopupInfo;

    RefPtr<WebCore::TextIndicator> linkTextIndicator;

    WebHitTestResultData();
    WebHitTestResultData(WebHitTestResultData&&) = default;
    WebHitTestResultData(const WebHitTestResultData&) = default;
    WebHitTestResultData& operator=(WebHitTestResultData&&) = default;
    WebHitTestResultData& operator=(const WebHitTestResultData&) = default;
    WebHitTestResultData(const WebCore::HitTestResult&, const String& toolTipText);
    WebHitTestResultData(const WebCore::HitTestResult&, bool includeImage);
    WebHitTestResultData(const String& absoluteImageURL, const String& absolutePDFURL, const String& absoluteLinkURL, const String& absoluteMediaURL, const String& linkLabel, const String& linkTitle, const String& linkSuggestedFilename, const String& imageSuggestedFilename, bool isContentEditable, const WebCore::IntRect& elementBoundingBox, const WebKit::WebHitTestResultData::IsScrollbar&, bool isSelected, bool isTextNode, bool isOverTextInsideFormControlElement, bool isDownloadableMedia, bool mediaIsInFullscreen, bool isActivePDFAnnotation, const WebKit::WebHitTestResultData::ElementType&, std::optional<FrameInfoData>&&, std::optional<WebCore::RemoteUserInputEventData>, const String& lookupText, const String& toolTipText, const String& imageText, std::optional<WebCore::SharedMemory::Handle>&& imageHandle, const RefPtr<WebCore::ShareableBitmap>& imageBitmap, const String& sourceImageMIMEType, const String& linkLocalDataMIMEType, bool hasLocalDataForLinkURL, bool hasEntireImage,
#if PLATFORM(MAC)
        const WebHitTestResultPlatformData&,
#endif
        const WebCore::DictionaryPopupInfo&, const RefPtr<WebCore::TextIndicator>&);
    ~WebHitTestResultData();

    WebCore::IntRect elementBoundingBoxInWindowCoordinates(const WebCore::HitTestResult&);

    static std::optional<FrameInfoData> frameInfoDataFromHitTestResult(const WebCore::HitTestResult&);

    std::optional<WebCore::SharedMemory::Handle> getImageSharedMemoryHandle() const;

};

} // namespace WebKit
