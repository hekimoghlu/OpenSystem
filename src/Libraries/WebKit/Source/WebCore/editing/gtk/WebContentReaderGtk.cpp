/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#include "WebContentReader.h"

#include "Blob.h"
#include "BlobURL.h"
#include "DOMURL.h"
#include "Document.h"
#include "DocumentFragment.h"
#include "Editor.h"
#include "LocalFrame.h"
#include "Page.h"
#include "Settings.h"
#include "markup.h"
#include <wtf/URL.h>

namespace WebCore {

bool WebContentReader::readFilePath(const String& path, PresentationSize, const String&)
{
    if (path.isEmpty() || !frame().document())
        return false;

    auto markup = urlToMarkup(URL({ }, path), path);
    addFragment(createFragmentFromMarkup(*frame().protectedDocument(), markup, "file://"_s, { }));

    return true;
}

bool WebContentReader::readHTML(const String& string)
{
    if (frame().settings().preferMIMETypeForImages() || !frame().document())
        return false;

    addFragment(createFragmentFromMarkup(*frame().protectedDocument(), string, emptyString(), { }));
    return true;
}

bool WebContentReader::readPlainText(const String& text)
{
    if (!m_allowPlainText)
        return false;

    addFragment(createFragmentFromText(m_context, text));

    m_madeFragmentFromPlainText = true;
    return true;
}

bool WebContentReader::readImage(Ref<FragmentedSharedBuffer>&& buffer, const String& type, PresentationSize preferredPresentationSize)
{
    ASSERT(frame().document());
    Ref document = *frame().document();
    addFragment(createFragmentForImageAndURL(document, DOMURL::createObjectURL(document, Blob::create(document.ptr(), buffer->extractData(), type)), preferredPresentationSize));

    return m_fragment;
}

bool WebContentReader::readURL(const URL&, const String&)
{
    return false;
}

static bool shouldReplaceSubresourceURL(const URL& url)
{
    return !(url.protocolIsInHTTPFamily() || url.protocolIsData());
}

bool WebContentMarkupReader::readHTML(const String& string)
{
    if (!frame().document())
        return false;

    if (shouldSanitize()) {
        m_markup = sanitizeMarkup(string, MSOListQuirks::Disabled, Function<void(DocumentFragment&)> { [](DocumentFragment& fragment) {
            removeSubresourceURLAttributes(fragment, [](const URL& url) {
                return shouldReplaceSubresourceURL(url);
            });
        } });
    } else
        m_markup = string;

    return !m_markup.isEmpty();
}

} // namespace WebCore
