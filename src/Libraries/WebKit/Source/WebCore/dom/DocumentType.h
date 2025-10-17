/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

#include "Node.h"

namespace WebCore {

class NamedNodeMap;

class DocumentType final : public Node {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DocumentType);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DocumentType);
public:
    static Ref<DocumentType> create(Document& document, const String& name, const String& publicId, const String& systemId)
    {
        return adoptRef(*new DocumentType(document, name, publicId, systemId));
    }

    const String& name() const { return m_name; }
    const String& publicId() const { return m_publicId; }
    const String& systemId() const { return m_systemId; }

private:
    DocumentType(Document&, const String& name, const String& publicId, const String& systemId);

    String nodeName() const override;
    Ref<Node> cloneNodeInternal(TreeScope&, CloningOperation) override;

    void parentOrShadowHostNode() const = delete; // Call parentNode() instead.

    String m_name;
    String m_publicId;
    String m_systemId;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::DocumentType)
    static bool isType(const WebCore::Node& node) { return node.nodeType() == WebCore::Node::DOCUMENT_TYPE_NODE; }
SPECIALIZE_TYPE_TRAITS_END()
