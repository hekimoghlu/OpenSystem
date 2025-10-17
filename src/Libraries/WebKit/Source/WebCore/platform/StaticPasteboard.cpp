/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#include "StaticPasteboard.h"

#include "CommonAtomStrings.h"
#include "SharedBuffer.h"

namespace WebCore {

StaticPasteboard::StaticPasteboard()
    : Pasteboard({ })
{
}

StaticPasteboard::~StaticPasteboard() = default;

bool StaticPasteboard::hasData()
{
    return m_customData.hasData();
}

Vector<String> StaticPasteboard::typesSafeForBindings(const String&)
{
    return m_customData.orderedTypes();
}

Vector<String> StaticPasteboard::typesForLegacyUnsafeBindings()
{
    return m_customData.orderedTypes();
}

String StaticPasteboard::readString(const String& type)
{
    return m_customData.readString(type);
}

String StaticPasteboard::readStringInCustomData(const String& type)
{
    return m_customData.readStringInCustomData(type);
}

bool StaticPasteboard::hasNonDefaultData() const
{
    return !m_nonDefaultDataTypes.isEmpty();
}

void StaticPasteboard::writeString(const String& type, const String& value)
{
    m_nonDefaultDataTypes.add(type);
    m_customData.writeString(type, value);
}

void StaticPasteboard::writeData(const String& type, Ref<SharedBuffer>&& data)
{
    m_nonDefaultDataTypes.add(type);
    m_customData.writeData(type, WTFMove(data));
}

void StaticPasteboard::writeStringInCustomData(const String& type, const String& value)
{
    m_nonDefaultDataTypes.add(type);
    m_customData.writeStringInCustomData(type, value);
}

void StaticPasteboard::clear()
{
    m_nonDefaultDataTypes.clear();
    m_fileContentState = Pasteboard::FileContentState::NoFileOrImageData;
    m_customData.clear();
}

void StaticPasteboard::clear(const String& type)
{
    m_nonDefaultDataTypes.remove(type);
    m_customData.clear(type);
}

PasteboardCustomData StaticPasteboard::takeCustomData()
{
    return std::exchange(m_customData, { });
}

void StaticPasteboard::writeMarkup(const String& markup)
{
    m_customData.writeString(textHTMLContentTypeAtom(), markup);
}

void StaticPasteboard::writePlainText(const String& text, SmartReplaceOption)
{
    m_customData.writeString(textPlainContentTypeAtom(), text);
}

void StaticPasteboard::write(const PasteboardURL& url)
{
    m_customData.writeString("text/uri-list"_s, url.url.string());
}

void StaticPasteboard::write(const PasteboardImage& image)
{
    // FIXME: This should ideally remember the image data, so that when this StaticPasteboard
    // is committed to the native pasteboard, we'll preserve the image as well. For now, stick
    // with our existing behavior, which prevents image data from being copied in the case where
    // any non-default data was written by the page.
    m_fileContentState = Pasteboard::FileContentState::InMemoryImage;

#if PLATFORM(MAC)
    if (!image.dataInHTMLFormat.isEmpty())
        writeMarkup(image.dataInHTMLFormat);
#else
    UNUSED_PARAM(image);
#endif

#if !PLATFORM(WIN)
    if (Pasteboard::canExposeURLToDOMWhenPasteboardContainsFiles(image.url.url.string()))
        write(image.url);
#endif
}

void StaticPasteboard::write(const PasteboardWebContent& content)
{
    String markup;
    String text;

#if PLATFORM(COCOA)
    markup = content.dataInHTMLFormat;
    text = content.dataInStringFormat;
#elif PLATFORM(GTK) || USE(LIBWPE)
    markup = content.markup;
    text = content.text;
#else
    UNUSED_PARAM(content);
#endif

    if (!markup.isEmpty())
        writeMarkup(markup);

    if (!text.isEmpty())
        writePlainText(text, SmartReplaceOption::CannotSmartReplace);
}

}
