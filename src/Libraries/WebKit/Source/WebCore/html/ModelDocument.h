/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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

#if ENABLE(MODEL_ELEMENT)

#include "HTMLDocument.h"

namespace WebCore {

class ModelDocument final : public HTMLDocument {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ModelDocument);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ModelDocument);
public:
    static Ref<ModelDocument> create(LocalFrame* frame, const Settings& settings, const URL& url)
    {
        auto document = adoptRef(*new ModelDocument(frame, settings, url));
        document->addToContextsMap();
        return document;
    }

    virtual ~ModelDocument() = default;

    String outgoingReferrer() const { return m_outgoingReferrer; }

private:
    ModelDocument(LocalFrame*, const Settings&, const URL&);

    Ref<DocumentParser> createParser() override;

    String m_outgoingReferrer;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ModelDocument)
    static bool isType(const WebCore::Document& document) { return document.isModelDocument(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* document = dynamicDowncast<WebCore::Document>(node);
        return document && isType(*document);
    }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MODEL_ELEMENT)
