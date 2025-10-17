/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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

#include "Node.h"
#include "XSLStyleSheet.h"
// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
IGNORE_WARNINGS_BEGIN("undef")
#include <libxml/parserInternals.h>
#include <libxslt/documents.h>
IGNORE_WARNINGS_END
IGNORE_WARNINGS_END
#include <wtf/HashMap.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class Document;
class DocumentFragment;
class LocalFrame;

class XSLTProcessor : public RefCounted<XSLTProcessor> {
public:
    static Ref<XSLTProcessor> create() { return adoptRef(*new XSLTProcessor); }
    ~XSLTProcessor();

    void setXSLStyleSheet(RefPtr<XSLStyleSheet>&& styleSheet) { m_stylesheet = WTFMove(styleSheet); }
    bool transformToString(Node& source, String& resultMIMEType, String& resultString, String& resultEncoding);
    Ref<Document> createDocumentFromSource(const String& source, const String& sourceEncoding, const String& sourceMIMEType, Node* sourceNode, LocalFrame*);
    
    // DOM methods
    void importStylesheet(Ref<Node>&& style)
    {
        m_stylesheetRootNode = WTFMove(style);
    }
    RefPtr<DocumentFragment> transformToFragment(Node& source, Document& ouputDocument);
    RefPtr<Document> transformToDocument(Node& source);
    
    void setParameter(const String& namespaceURI, const String& localName, const String& value);
    String getParameter(const String& namespaceURI, const String& localName) const;
    void removeParameter(const String& namespaceURI, const String& localName);
    void clearParameters() { m_parameters.clear(); }

    void reset();

#if LIBXML_VERSION >= 21200
    static void parseErrorFunc(void* userData, const xmlError*);
#else
    static void parseErrorFunc(void* userData, xmlError*);
#endif
    static void genericErrorFunc(void* userData, const char* msg, ...);
    
    // Only for libXSLT callbacks
    XSLStyleSheet* xslStylesheet() const { return m_stylesheet.get(); }

    using ParameterMap = HashMap<String, String>;

private:
    XSLTProcessor() = default;

    RefPtr<XSLStyleSheet> m_stylesheet;
    RefPtr<Node> m_stylesheetRootNode;
    ParameterMap m_parameters;
};

} // namespace WebCore

#endif // ENABLE(XSLT)
