/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#import "HistoryPropertyList.h"

#import "WebHistoryItemInternal.h"
#import <WebCore/HistoryItem.h>

using namespace WebCore;

static const int currentFileVersion = 1;

HistoryPropertyListWriter::HistoryPropertyListWriter()
    : m_displayTitleKey("displayTitle"_s)
    , m_lastVisitWasFailureKey("lastVisitWasFailure"_s)
    , m_lastVisitedDateKey("lastVisitedDate"_s)
    , m_redirectURLsKey("redirectURLs"_s)
    , m_titleKey("title"_s)
    , m_urlKey(emptyString())
    , m_buffer(0)
{
}

UInt8* HistoryPropertyListWriter::buffer(size_t size)
{
    ASSERT(!m_buffer);
    m_buffer = static_cast<UInt8*>(CFAllocatorAllocate(0, size, 0));
    m_bufferSize = size;
    return m_buffer;
}

RetainPtr<CFDataRef> HistoryPropertyListWriter::releaseData()
{
    UInt8* buffer = m_buffer;
    if (!buffer)
        return 0;
    m_buffer = 0;
    RetainPtr<CFDataRef> data = adoptCF(CFDataCreateWithBytesNoCopy(0, buffer, m_bufferSize, 0));
    if (!data) {
        CFAllocatorDeallocate(0, buffer);
        return 0;
    }
    return data;
}

void HistoryPropertyListWriter::writeObjects(BinaryPropertyListObjectStream& stream)
{
    size_t outerDictionaryStart = stream.writeDictionaryStart();

    stream.writeString("WebHistoryFileVersion"_s);
    stream.writeString("WebHistoryDates"_s);

    stream.writeInteger(currentFileVersion);
    size_t outerDateArrayStart = stream.writeArrayStart();
    writeHistoryItems(stream);
    stream.writeArrayEnd(outerDateArrayStart);

    stream.writeDictionaryEnd(outerDictionaryStart);
}

void HistoryPropertyListWriter::writeHistoryItem(BinaryPropertyListObjectStream& stream, WebHistoryItem* webHistoryItem)
{
    HistoryItem* item = core(webHistoryItem);

    size_t itemDictionaryStart = stream.writeDictionaryStart();

    const String& title = item->title();
    const String& displayTitle = item->alternateTitle();
    double lastVisitedDate = webHistoryItem->_private->_lastVisitedTime;
    Vector<String>* redirectURLs = webHistoryItem->_private->_redirectURLs.get();

    // keys
    stream.writeString(m_urlKey);
    if (!title.isEmpty())
        stream.writeString(m_titleKey);
    if (!displayTitle.isEmpty())
        stream.writeString(m_displayTitleKey);
    if (lastVisitedDate)
        stream.writeString(m_lastVisitedDateKey);
    if (item->lastVisitWasFailure())
        stream.writeString(m_lastVisitWasFailureKey);
    if (redirectURLs)
        stream.writeString(m_redirectURLsKey);

    // values
    stream.writeUniqueString(item->urlString());
    if (!title.isEmpty())
        stream.writeString(title);
    if (!displayTitle.isEmpty())
        stream.writeString(displayTitle);
    if (lastVisitedDate) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.1lf", lastVisitedDate);
        stream.writeUniqueString(buffer);
    }
    if (item->lastVisitWasFailure())
        stream.writeBooleanTrue();
    if (redirectURLs) {
        size_t redirectArrayStart = stream.writeArrayStart();
        size_t size = redirectURLs->size();
        ASSERT(size);
        for (size_t i = 0; i < size; ++i)
            stream.writeUniqueString(redirectURLs->at(i));
        stream.writeArrayEnd(redirectArrayStart);
    }

    stream.writeDictionaryEnd(itemDictionaryStart);
}

