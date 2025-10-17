/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
#include "config.h"
#include "TextDocumentParser.h"

#include "CSSTokenizerInputStream.h"
#include "HTMLTreeBuilder.h"
#include "ScriptElement.h"

namespace WebCore {

using namespace HTMLNames;

TextDocumentParser::TextDocumentParser(HTMLDocument& document)
    : HTMLDocumentParser(document)
{
}

void TextDocumentParser::append(RefPtr<StringImpl>&& text)
{
    if (!m_hasInsertedFakeFormattingElements)
        insertFakeFormattingElements();
    HTMLDocumentParser::append(WTFMove(text));
}

void TextDocumentParser::insertFakeFormattingElements()
{
    // In principle, we should create a specialized tree builder for
    // TextDocuments, but instead we re-use the existing HTMLTreeBuilder.
    // We create fake tokens and give it to the tree builder rather than
    // sending fake bytes through the front-end of the parser to avoid
    // distrubing the line/column number calculations.

    Attribute nameAttribute(nameAttr, "color-scheme"_s);
    Attribute contentAttribute(contentAttr, "light dark"_s);
    AtomHTMLToken fakeMeta(HTMLToken::Type::StartTag, TagName::meta, { WTFMove(nameAttribute), WTFMove(contentAttribute) });
    treeBuilder().constructTree(WTFMove(fakeMeta));

    Attribute attribute(styleAttr, "word-wrap: break-word; white-space: pre-wrap;"_s);
    AtomHTMLToken fakePre(HTMLToken::Type::StartTag, TagName::pre, { WTFMove(attribute) });
    treeBuilder().constructTree(WTFMove(fakePre));

    // Normally we would skip the first \n after a <pre> element, but we don't
    // want to skip the first \n for text documents!
    treeBuilder().setShouldSkipLeadingNewline(false);

    // Although Text Documents expose a "pre" element in their DOM, they
    // act like a <plaintext> tag, so we have to force plaintext mode.
    tokenizer().setPLAINTEXTState();

    m_hasInsertedFakeFormattingElements = true;
}

}
