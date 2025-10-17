/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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

#if ENABLE(CONTEXT_MENUS)

#include "WebContextMenuItemData.h"
#include "WebHitTestResultData.h"
#include <WebCore/ContextMenuContext.h>
#include <WebCore/ElementContext.h>

#if ENABLE(SERVICE_CONTROLS)
#include <WebCore/AttributedString.h>
#endif

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

class ContextMenuContextData {
public:
    using Type = WebCore::ContextMenuContext::Type;

    ContextMenuContextData();
    ContextMenuContextData(const WebCore::IntPoint& menuLocation, const Vector<WebKit::WebContextMenuItemData>& menuItems, const WebCore::ContextMenuContext&);

    ContextMenuContextData(WebCore::ContextMenuContext::Type
        , WebCore::IntPoint&& menuLocation
        , Vector<WebContextMenuItemData>&& menuItems
        , std::optional<WebKit::WebHitTestResultData>&&
        , String&& selectedText
#if ENABLE(SERVICE_CONTROLS)
        , std::optional<WebCore::ShareableBitmapHandle>&& controlledImageHandle
        , WebCore::AttributedString&& controlledSelection
        , Vector<String>&& selectedTelephoneNumbers
        , bool selectionIsEditable
        , WebCore::IntRect&& controlledImageBounds
        , String&& controlledImageAttachmentID
        , std::optional<WebCore::ElementContext>&& controlledImageElementContext
        , String&& controlledImageMIMEType
#endif // ENABLE(SERVICE_CONTROLS)
#if ENABLE(CONTEXT_MENU_QR_CODE_DETECTION)
        , std::optional<WebCore::ShareableBitmapHandle>&& potentialQRCodeNodeSnapshotImageHandle
        , std::optional<WebCore::ShareableBitmapHandle>&& potentialQRCodeViewportSnapshotImageHandle
#endif // ENABLE(CONTEXT_MENU_QR_CODE_DETECTION)
        , bool hasEntireImage
    );

    Type type() const { return m_type; }
    const WebCore::IntPoint& menuLocation() const { return m_menuLocation; }
    void setMenuLocation(WebCore::IntPoint menuLocation) { m_menuLocation = menuLocation; }
    const Vector<WebKit::WebContextMenuItemData>& menuItems() const { return m_menuItems; }

    const std::optional<WebHitTestResultData>& webHitTestResultData() const { return m_webHitTestResultData; }
    const String& selectedText() const { return m_selectedText; }

    bool hasEntireImage() const { return m_hasEntireImage; }

#if ENABLE(SERVICE_CONTROLS)
    ContextMenuContextData(const WebCore::IntPoint& menuLocation, WebCore::AttributedString&& controlledSelection, const Vector<String>& selectedTelephoneNumbers, bool isEditable)
        : m_type(Type::ServicesMenu)
        , m_menuLocation(menuLocation)
        , m_controlledSelection(WTFMove(controlledSelection))
        , m_selectedTelephoneNumbers(selectedTelephoneNumbers)
        , m_selectionIsEditable(isEditable)
    {
    }

    ContextMenuContextData(const WebCore::IntPoint& menuLocation, bool isEditable, const WebCore::IntRect& imageRect, const String& attachmentID, const String& sourceImageMIMEType)
        : m_type(Type::ServicesMenu)
        , m_menuLocation(menuLocation)
        , m_selectionIsEditable(isEditable)
        , m_controlledImageBounds(imageRect)
        , m_controlledImageAttachmentID(attachmentID)
        , m_controlledImageMIMEType(sourceImageMIMEType)
    {
    }

    ContextMenuContextData(const WebCore::IntPoint& menuLocation, WebCore::Image&, bool isEditable, const WebCore::IntRect& imageRect, const String& attachmentID, std::optional<WebCore::ElementContext>&&, const String& sourceImageMIMEType);

    WebCore::ShareableBitmap* controlledImage() const { return m_controlledImage.get(); }
    std::optional<WebCore::ShareableBitmap::Handle> createControlledImageReadOnlyHandle() const;

    const WebCore::AttributedString& controlledSelection() const { return m_controlledSelection; }
    const Vector<String>& selectedTelephoneNumbers() const { return m_selectedTelephoneNumbers; }

    bool selectionIsEditable() const { return m_selectionIsEditable; }

    bool isServicesMenu() const { return m_type == ContextMenuContextData::Type::ServicesMenu; }
    bool controlledDataIsEditable() const;
    WebCore::IntRect controlledImageBounds() const { return m_controlledImageBounds; };
    String controlledImageAttachmentID() const { return m_controlledImageAttachmentID; };
    std::optional<WebCore::ElementContext> controlledImageElementContext() const { return m_controlledImageElementContext; }
    String controlledImageMIMEType() const { return m_controlledImageMIMEType; }
#endif // ENABLE(SERVICE_CONTROLS)

#if ENABLE(CONTEXT_MENU_QR_CODE_DETECTION)
    WebCore::ShareableBitmap* potentialQRCodeNodeSnapshotImage() const { return m_potentialQRCodeNodeSnapshotImage.get(); }
    std::optional<WebCore::ShareableBitmap::Handle> createPotentialQRCodeNodeSnapshotImageReadOnlyHandle() const;
    WebCore::ShareableBitmap* potentialQRCodeViewportSnapshotImage() const { return m_potentialQRCodeViewportSnapshotImage.get(); }
    std::optional<WebCore::ShareableBitmap::Handle> createPotentialQRCodeViewportSnapshotImageReadOnlyHandle() const;

    const String& qrCodePayloadString() const { return m_qrCodePayloadString; }
    void setQRCodePayloadString(const String& string) { m_qrCodePayloadString = string; }
#endif

private:
    Type m_type;

    WebCore::IntPoint m_menuLocation;
    Vector<WebKit::WebContextMenuItemData> m_menuItems;

    std::optional<WebHitTestResultData> m_webHitTestResultData;
    String m_selectedText;
    bool m_hasEntireImage { false };

#if ENABLE(SERVICE_CONTROLS)
    void setImage(WebCore::Image&);
    
    RefPtr<WebCore::ShareableBitmap> m_controlledImage;
    WebCore::AttributedString m_controlledSelection;
    Vector<String> m_selectedTelephoneNumbers;
    bool m_selectionIsEditable;
    WebCore::IntRect m_controlledImageBounds;
    String m_controlledImageAttachmentID;
    std::optional<WebCore::ElementContext> m_controlledImageElementContext;
    String m_controlledImageMIMEType;
#endif

#if ENABLE(CONTEXT_MENU_QR_CODE_DETECTION)
    void setPotentialQRCodeNodeSnapshotImage(WebCore::Image&);
    void setPotentialQRCodeViewportSnapshotImage(WebCore::Image&);

    RefPtr<WebCore::ShareableBitmap> m_potentialQRCodeNodeSnapshotImage;
    RefPtr<WebCore::ShareableBitmap> m_potentialQRCodeViewportSnapshotImage;

    String m_qrCodePayloadString;
#endif
};

} // namespace WebKit

#endif // ENABLE(CONTEXT_MENUS)
