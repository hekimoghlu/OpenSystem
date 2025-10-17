/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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

#include "HitTestResult.h"
#include "Image.h"

namespace WebCore {

class Event;

enum class ContextMenuContextType : uint8_t {
    ContextMenu,
#if ENABLE(SERVICE_CONTROLS)
    ServicesMenu,
#endif // ENABLE(SERVICE_CONTROLS)
#if ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS)
    MediaControls,
#endif // ENABLE(MEDIA_CONTROLS_CONTEXT_MENUS)
};

class ContextMenuContext {
public:
    using Type = ContextMenuContextType;

    ContextMenuContext();
    ContextMenuContext(Type, const HitTestResult&, RefPtr<Event>&&);

    ~ContextMenuContext();

    ContextMenuContext& operator=(const ContextMenuContext&);

    Type type() const { return m_type; }

    const HitTestResult& hitTestResult() const { return m_hitTestResult; }
    Event* event() const { return m_event.get(); }

    void setSelectedText(const String& selectedText) { m_selectedText = selectedText; }
    const String& selectedText() const { return m_selectedText; }

    bool hasEntireImage() const { return m_hasEntireImage; }

#if ENABLE(SERVICE_CONTROLS)
    void setControlledImage(Image* controlledImage) { m_controlledImage = controlledImage; }
    Image* controlledImage() const { return m_controlledImage.get(); }
#endif

#if ENABLE(CONTEXT_MENU_QR_CODE_DETECTION)
    void setPotentialQRCodeNodeSnapshotImage(Image* image) { m_potentialQRCodeNodeSnapshotImage = image; }
    Image* potentialQRCodeNodeSnapshotImage() const { return m_potentialQRCodeNodeSnapshotImage.get(); }

    void setPotentialQRCodeViewportSnapshotImage(Image* image) { m_potentialQRCodeViewportSnapshotImage = image; }
    Image* potentialQRCodeViewportSnapshotImage() const { return m_potentialQRCodeViewportSnapshotImage.get(); }
#endif

private:
    Type m_type { Type::ContextMenu };
    HitTestResult m_hitTestResult;
    RefPtr<Event> m_event;
    String m_selectedText;
    bool m_hasEntireImage { false };

#if ENABLE(SERVICE_CONTROLS)
    RefPtr<Image> m_controlledImage;
#endif

#if ENABLE(CONTEXT_MENU_QR_CODE_DETECTION)
    RefPtr<Image> m_potentialQRCodeNodeSnapshotImage;
    RefPtr<Image> m_potentialQRCodeViewportSnapshotImage;
#endif
};

} // namespace WebCore

#endif // ENABLE(CONTEXT_MENUS)
