/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#include "ContentExtensionStyleSheet.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include "CSSStyleSheet.h"
#include "ContentExtensionParser.h"
#include "ContentExtensionsBackend.h"
#include "Document.h"
#include "StyleSheetContents.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace ContentExtensions {

ContentExtensionStyleSheet::ContentExtensionStyleSheet(Document& document)
    : m_styleSheet(CSSStyleSheet::create(StyleSheetContents::create(contentExtensionCSSParserContext()), document))
{
    m_styleSheet->contents().setIsUserStyleSheet(true);
}

ContentExtensionStyleSheet::~ContentExtensionStyleSheet()
{
    m_styleSheet->clearOwnerNode();
}

bool ContentExtensionStyleSheet::addDisplayNoneSelector(const String& selector, uint32_t selectorID)
{
    ASSERT(selectorID != std::numeric_limits<uint32_t>::max());

    if (!m_addedSelectorIDs.add(selectorID).isNewEntry)
        return false;

    StringBuilder css;
    css.append(selector);
    css.append('{');
    css.append(ContentExtensionsBackend::displayNoneCSSRule());
    css.append('}');
    m_styleSheet->contents().parseString(css.toString());
    return true;
}

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
