/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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

#include "Document.h"

namespace WebCore {

class XMLDocument : public Document {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XMLDocument);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(XMLDocument);
public:
    static Ref<XMLDocument> create(LocalFrame* frame, const Settings& settings, const URL& url)
    {
        auto document = adoptRef(*new XMLDocument(frame, settings, url, { DocumentClass::XML }));
        document->addToContextsMap();
        return document;
    }

    WEBCORE_EXPORT static Ref<XMLDocument> createXHTML(LocalFrame*, const Settings&, const URL&);

protected:
    XMLDocument(LocalFrame* frame, const Settings& settings, const URL& url, DocumentClasses documentClasses = { })
        : Document(frame, settings, url, documentClasses | DocumentClasses(DocumentClass::XML))
    {
    }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::XMLDocument)
    static bool isType(const WebCore::Document& document) { return document.isXMLDocument(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* document = dynamicDowncast<WebCore::Document>(node);
        return document && isType(*document);
    }
SPECIALIZE_TYPE_TRAITS_END()
