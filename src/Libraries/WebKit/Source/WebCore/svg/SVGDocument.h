/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

#include "XMLDocument.h"

namespace WebCore {

class SVGSVGElement;

class SVGDocument final : public XMLDocument {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGDocument);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGDocument);
public:
    static Ref<SVGDocument> create(LocalFrame*, const Settings&, const URL&);

    bool zoomAndPanEnabled() const;
    void startPan(const FloatPoint& start);
    void updatePan(const FloatPoint& position) const;

private:
    SVGDocument(LocalFrame*, const Settings&, const URL&);

    Ref<Document> cloneDocumentWithoutChildren() const override;

    FloatSize m_panningOffset;
};

inline Ref<SVGDocument> SVGDocument::create(LocalFrame* frame, const Settings& settings, const URL& url)
{
    Ref document = adoptRef(*new SVGDocument(frame, settings, url));
    document->addToContextsMap();
    return document;
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SVGDocument)
    static bool isType(const WebCore::Document& document) { return document.isSVGDocument(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* document = dynamicDowncast<WebCore::Document>(node);
        return document && isType(*document);
    }
SPECIALIZE_TYPE_TRAITS_END()
