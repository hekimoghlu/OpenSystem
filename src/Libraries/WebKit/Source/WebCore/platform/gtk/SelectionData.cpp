/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "SelectionData.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SelectionData);

SelectionData::SelectionData(const String& text, const String& markup, const URL& url, const String& uriList, RefPtr<WebCore::Image>&& image, RefPtr<WebCore::SharedBuffer>&& buffer, bool canSmartReplace)
{
    if (!text.isEmpty())
        setText(text);
    if (!markup.isEmpty())
        setMarkup(markup);
    if (!url.isEmpty())
        setURL(url, String());
    if (!uriList.isEmpty())
        setURIList(uriList);
    if (image)
        setImage(WTFMove(image));
    if (buffer)
        setCustomData(buffer.releaseNonNull());
    setCanSmartReplace(canSmartReplace);
}

static void replaceNonBreakingSpaceWithSpace(String& string)
{
    string = makeStringByReplacingAll(string, noBreakSpace, space);
}

void SelectionData::setText(const String& newText)
{
    m_text = newText;
    replaceNonBreakingSpaceWithSpace(m_text);
}

void SelectionData::setURIList(const String& uriListString)
{
    m_uriList = uriListString;

    // This code is originally from: platform/chromium/ChromiumDataObject.cpp.
    // FIXME: We should make this code cross-platform eventually.

    // Line separator is \r\n per RFC 2483 - however, for compatibility
    // reasons we also allow just \n here.

    // Process the input and copy the first valid URL into the url member.
    // In case no URLs can be found, subsequent calls to getData("URL")
    // will get an empty string. This is in line with the HTML5 spec (see
    // "The DragEvent and DataTransfer interfaces"). Also extract all filenames
    // from the URI list.
    bool setURL = hasURL();
    for (auto& line : uriListString.split('\n')) {
        line = line.trim(deprecatedIsSpaceOrNewline);
        if (line.isEmpty())
            continue;
        if (line[0] == '#')
            continue;

        URL url { line };
        if (url.isValid()) {
            if (!setURL) {
                m_url = url;
                setURL = true;
            }

            GUniqueOutPtr<GError> error;
            GUniquePtr<gchar> filename(g_filename_from_uri(line.utf8().data(), 0, &error.outPtr()));
            if (!error && filename)
                m_filenames.append(String::fromUTF8(filename.get()));
        }
    }
}

void SelectionData::setURL(const URL& url, const String& label)
{
    m_url = url;
    if (m_uriList.isEmpty())
        m_uriList = url.string();

    if (!hasText())
        setText(url.string());

    if (hasMarkup())
        return;

    String actualLabel = label.isEmpty() ? url.string() : label;
    GUniquePtr<gchar> escaped(g_markup_escape_text(actualLabel.utf8().data(), -1));

    setMarkup(makeString("<a href=\""_s, url.string(), "\">"_s,
        String::fromUTF8(escaped.get()), "</a>"_s));
}

const String& SelectionData::urlLabel() const
{
    if (hasText())
        return text();

    if (hasURL())
        return url().string();

    return emptyString();
}

void SelectionData::clearAllExceptFilenames()
{
    clearText();
    clearMarkup();
    clearURIList();
    clearURL();
    clearImage();
    clearCustomData();
    clearBuffers();

    m_canSmartReplace = false;
}

void SelectionData::clearAll()
{
    clearAllExceptFilenames();
    m_filenames.clear();
}

} // namespace WebCore
