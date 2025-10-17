/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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

#include "HTMLDocument.h"

namespace WebCore {

class ImageDocumentElement;
class HTMLImageElement;

class ImageDocument final : public HTMLDocument {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageDocument);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ImageDocument);
public:
    static Ref<ImageDocument> create(LocalFrame& frame, const URL& url)
    {
        auto document = adoptRef(*new ImageDocument(frame, url));
        document->addToContextsMap();
        return document;
    }

    WEBCORE_EXPORT HTMLImageElement* imageElement() const;

    void updateDuringParsing();
    void finishedParsing();

    void disconnectImageElement() { m_imageElement = nullptr; }

#if !PLATFORM(IOS_FAMILY)
    void imageClicked(int x, int y);
#endif

private:
    ImageDocument(LocalFrame&, const URL&);

    Ref<DocumentParser> createParser() override;

    LayoutSize imageSize();

    void createDocumentStructure();
#if !PLATFORM(IOS_FAMILY)
    void resizeImageToFit();
    void restoreImageSize();
    bool imageFitsInWindow();
    float scale();
    void didChangeViewSize() final;
#endif

    void imageUpdated();

    WeakPtr<ImageDocumentElement, WeakPtrImplWithEventTargetData> m_imageElement;

    // Whether enough of the image has been loaded to determine its size.
    bool m_imageSizeIsKnown;

#if !PLATFORM(IOS_FAMILY)
    // Whether the image is shrunk to fit or not.
    bool m_didShrinkImage;
#endif

    // Whether the image should be shrunk or not.
    bool m_shouldShrinkImage;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ImageDocument)
    static bool isType(const WebCore::Document& document) { return document.isImageDocument(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* document = dynamicDowncast<WebCore::Document>(node);
        return document && isType(*document);
    }
SPECIALIZE_TYPE_TRAITS_END()
