/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

#include "CSSStyleSheet.h"
#include "StyleScope.h"
#include <wtf/text/TextPosition.h>

namespace WebCore {

class Document;
class Element;

class InlineStyleSheetOwner {
public:
    InlineStyleSheetOwner(Document&, bool createdByParser);
    ~InlineStyleSheetOwner();

    void setContentType(const AtomString& contentType) { m_contentType = contentType; }
    void setMedia(const AtomString& media) { m_media = media; }

    CSSStyleSheet* sheet() const { return m_sheet.get(); }

    bool isLoading() const;
    bool sheetLoaded(Element&);
    void startLoadingDynamicSheet(Element&);

    void insertedIntoDocument(Element&);
    void removedFromDocument(Element&);
    void clearDocumentData(Element&);
    void childrenChanged(Element&);
    void finishParsingChildren(Element&);

    Style::Scope* styleScope() { return m_styleScope.get(); }

private:
    void createSheet(Element&, const String& text);
    void createSheetFromTextContents(Element&);
    void clearSheet();

    bool m_isParsingChildren;
    bool m_loading;
    TextPosition m_startTextPosition;
    AtomString m_contentType;
    AtomString m_media;
    RefPtr<CSSStyleSheet> m_sheet;
    WeakPtr<Style::Scope> m_styleScope;
};

} // namespace WebCore
