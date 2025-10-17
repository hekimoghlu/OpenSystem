/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

#include "CachedResourceHandle.h"
#include "CachedStyleSheetClient.h"
#include "CharacterData.h"

namespace WebCore {

class StyleSheet;
class CSSStyleSheet;

class ProcessingInstruction final : public CharacterData, private CachedStyleSheetClient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ProcessingInstruction);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ProcessingInstruction);
public:
    USING_CAN_MAKE_WEAKPTR(CharacterData);

    static Ref<ProcessingInstruction> create(Document&, String&& target, String&& data);
    virtual ~ProcessingInstruction();

    const String& target() const { return m_target; }

    void setCreatedByParser(bool createdByParser) { m_createdByParser = createdByParser; }

    const String& localHref() const { return m_localHref; }
    StyleSheet* sheet() const { return m_sheet.get(); }
    RefPtr<StyleSheet> protectedSheet() const;

    bool isCSS() const { return m_isCSS; }
#if ENABLE(XSLT)
    bool isXSL() const { return m_isXSL; }
#endif

private:
    friend class CharacterData;
    ProcessingInstruction(Document&, String&& target, String&& data);

    String nodeName() const override;
    Ref<Node> cloneNodeInternal(TreeScope&, CloningOperation) override;

    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) override;
    void didFinishInsertingNode() override;
    void removedFromAncestor(RemovalType, ContainerNode&) override;

    void checkStyleSheet();
    void setCSSStyleSheet(const String& href, const URL& baseURL, ASCIILiteral charset, const CachedCSSStyleSheet*) override;
#if ENABLE(XSLT)
    void setXSLStyleSheet(const String& href, const URL& baseURL, const String& sheet) override;
#endif

    bool isLoading() const;
    bool sheetLoaded() override;

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const override;

    void parseStyleSheet(const String& sheet);

    String m_target;
    String m_localHref;
    String m_title;
    String m_media;
    CachedResourceHandle<CachedResource> m_cachedSheet;
    RefPtr<StyleSheet> m_sheet;
    bool m_loading { false };
    bool m_alternate { false };
    bool m_createdByParser { false };
    bool m_isCSS { false };
#if ENABLE(XSLT)
    bool m_isXSL { false };
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ProcessingInstruction)
    static bool isType(const WebCore::Node& node) { return node.nodeType() == WebCore::Node::PROCESSING_INSTRUCTION_NODE; }
SPECIALIZE_TYPE_TRAITS_END()
