/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

#if ENABLE(XSLT)

#include "ProcessingInstruction.h"
#include "StyleSheet.h"
// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
IGNORE_WARNINGS_BEGIN("undef")
#include <libxml/parser.h>
#include <libxslt/transform.h>
IGNORE_WARNINGS_END
IGNORE_WARNINGS_END
#include <wtf/Ref.h>
#include <wtf/TypeCasts.h>

namespace WebCore {

class CachedResourceLoader;
class XSLImportRule;
    
class XSLStyleSheet final : public StyleSheet, public CanMakeWeakPtr<XSLStyleSheet> {
public:
    static Ref<XSLStyleSheet> create(XSLStyleSheet* parentSheet, const String& originalURL, const URL& finalURL)
    {
        return adoptRef(*new XSLStyleSheet(parentSheet, originalURL, finalURL));
    }

    static Ref<XSLStyleSheet> create(ProcessingInstruction& parentNode, const String& originalURL, const URL& finalURL)
    {
        return adoptRef(*new XSLStyleSheet(&parentNode, originalURL, finalURL, false));
    }

    static Ref<XSLStyleSheet> createEmbedded(ProcessingInstruction& parentNode, const URL& finalURL)
    {
        return adoptRef(*new XSLStyleSheet(&parentNode, finalURL.string(), finalURL, true));
    }

    // Taking an arbitrary node is unsafe, because owner node pointer can become stale.
    // XSLTProcessor ensures that the stylesheet doesn't outlive its parent, in part by not exposing it to JavaScript.
    static Ref<XSLStyleSheet> createForXSLTProcessor(Node* parentNode, const String& originalURL, const URL& finalURL)
    {
        return adoptRef(*new XSLStyleSheet(parentNode, originalURL, finalURL, false));
    }

    virtual ~XSLStyleSheet();

    bool parseString(const String&);
    
    void checkLoaded();
    
    const URL& finalURL() const { return m_finalURL; }

    void loadChildSheets();
    void loadChildSheet(const String& href);

    CachedResourceLoader* cachedResourceLoader();

    Document* ownerDocument();
    XSLStyleSheet* parentStyleSheet() const override { return m_parentStyleSheet.get(); }
    void setParentStyleSheet(XSLStyleSheet* parent);

    xmlDocPtr document();
    xsltStylesheetPtr compileStyleSheet();
    xmlDocPtr locateStylesheetSubResource(xmlDocPtr parentDoc, const xmlChar* uri);

    void clearDocuments();

    void markAsProcessed();
    bool processed() const { return m_processed; }
    
    String type() const override { return "text/xml"_s; }
    bool disabled() const override { return m_isDisabled; }
    void setDisabled(bool b) override { m_isDisabled = b; }
    Node* ownerNode() const override { return m_ownerNode.get(); }
    String href() const override { return m_originalURL; }
    String title() const override { return { }; }

    void clearOwnerNode() override { m_ownerNode = nullptr; }
    URL baseURL() const override { return m_finalURL; }
    bool isLoading() const override;

private:
    XSLStyleSheet(Node* parentNode, const String& originalURL, const URL& finalURL, bool embedded);
    XSLStyleSheet(XSLStyleSheet* parentSheet, const String& originalURL, const URL& finalURL);

    bool isXSLStyleSheet() const override { return true; }
    String debugDescription() const final;

    void clearXSLStylesheetDocument();

    WeakPtr<Node, WeakPtrImplWithEventTargetData> m_ownerNode;
    String m_originalURL;
    URL m_finalURL;
    bool m_isDisabled { false };

    Vector<std::unique_ptr<XSLImportRule>> m_children;

    bool m_embedded;
    bool m_processed;

    xmlDocPtr m_stylesheetDoc { nullptr };
    bool m_stylesheetDocTaken { false };
    bool m_compilationFailed { false };

    WeakPtr<XSLStyleSheet> m_parentStyleSheet;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::XSLStyleSheet)
    static bool isType(const WebCore::StyleSheet& styleSheet) { return styleSheet.isXSLStyleSheet(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(XSLT)
