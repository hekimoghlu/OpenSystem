/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#ifndef HistoryPropertyList_h
#define HistoryPropertyList_h

#include "BinaryPropertyList.h"
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

@class WebHistoryItem;

class HistoryPropertyListWriter : public BinaryPropertyListWriter {
public:
    RetainPtr<CFDataRef> releaseData();

protected:
    HistoryPropertyListWriter();

    void writeHistoryItem(BinaryPropertyListObjectStream&, WebHistoryItem *);

private:
    virtual void writeHistoryItems(BinaryPropertyListObjectStream&) = 0;

    virtual void writeObjects(BinaryPropertyListObjectStream&);
    virtual UInt8* buffer(size_t);

    const String m_displayTitleKey;
    const String m_lastVisitWasFailureKey;
    const String m_lastVisitedDateKey;
    const String m_redirectURLsKey;
    const String m_titleKey;
    const String m_urlKey;

    UInt8* m_buffer;
    size_t m_bufferSize;
};

#endif // HistoryPropertyList_h
