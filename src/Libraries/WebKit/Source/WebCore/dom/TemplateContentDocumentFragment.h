/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#include "DocumentFragment.h"
#include "Element.h"

namespace WebCore {

class TemplateContentDocumentFragment final : public DocumentFragment {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TemplateContentDocumentFragment);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TemplateContentDocumentFragment);
public:
    static Ref<TemplateContentDocumentFragment> create(Document& document, const Element& host)
    {
        return adoptRef(*new TemplateContentDocumentFragment(document, host));
    }

    const Element* host() const { return m_host.get(); }
    void clearHost() { m_host = nullptr; }

private:
    TemplateContentDocumentFragment(Document& document, const Element& host)
        : DocumentFragment(document)
        , m_host(host)
    {
    }

    bool isTemplateContent() const override { return true; }

    WeakPtr<const Element, WeakPtrImplWithEventTargetData> m_host;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::TemplateContentDocumentFragment)
    static bool isType(const WebCore::Node& node)
    {
        auto* fragment = dynamicDowncast<WebCore::DocumentFragment>(node);
        return fragment && is<WebCore::TemplateContentDocumentFragment>(*fragment);
    }
    static bool isType(const WebCore::DocumentFragment& node) { return node.isTemplateContent(); }
SPECIALIZE_TYPE_TRAITS_END()
