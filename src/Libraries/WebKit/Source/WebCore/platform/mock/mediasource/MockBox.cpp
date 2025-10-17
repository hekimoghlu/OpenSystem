/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#include "MockBox.h"

#if ENABLE(MEDIA_SOURCE)

#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/DataView.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/Int8Array.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSGlobalObjectInlines.h>
#include <JavaScriptCore/TypedArrayInlines.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

MockBox::MockBox(ArrayBuffer* data)
{
    m_type = peekType(data);
    m_length = peekLength(data);
    ASSERT(m_length >= 8);
}

String MockBox::peekType(ArrayBuffer* data)
{
    StringBuilder builder;
    auto array = JSC::Int8Array::create(data, 0, 4);
    for (int i = 0; i < 4; ++i)
        builder.append(static_cast<char>(array->item(i)));
    return builder.toString();
}

size_t MockBox::peekLength(ArrayBuffer* data)
{
    auto view = JSC::DataView::create(data, 0, data->byteLength());
    return view->get<uint32_t>(4, true);
}

MockTrackBox::MockTrackBox(ArrayBuffer* data)
    : MockBox(data)
{
    ASSERT(m_length == 17);

    auto view = JSC::DataView::create(data, 0, data->byteLength());
    m_trackID = view->get<int32_t>(8, true);

    StringBuilder builder;
    auto array = JSC::Int8Array::create(data, 12, 4);
    for (int i = 0; i < 4; ++i)
        builder.append(static_cast<char>(array->item(i)));
    m_codec = builder.toAtomString();

    m_kind = static_cast<TrackKind>(view->get<uint8_t>(16, true));
}

const String& MockTrackBox::type()
{
    static NeverDestroyed<String> trak(MAKE_STATIC_STRING_IMPL("trak"));
    return trak;
}

MockInitializationBox::MockInitializationBox(ArrayBuffer* data)
    : MockBox(data)
{
    ASSERT(m_length >= 13);

    auto view = JSC::DataView::create(data, 0, data->byteLength());
    int32_t timeValue = view->get<int32_t>(8, true);
    int32_t timeScale = view->get<int32_t>(12, true);
    m_duration = MediaTime(timeValue, timeScale);
    
    size_t offset = 16;

    while (offset < m_length) {
        auto subBuffer = data->slice(offset);
        if (MockBox::peekType(subBuffer.get()) != MockTrackBox::type())
            break;

        MockTrackBox trackBox(subBuffer.get());
        offset += trackBox.length();
        m_tracks.append(trackBox);
    }
}

const String& MockInitializationBox::type()
{
    static NeverDestroyed<String> init(MAKE_STATIC_STRING_IMPL("init"));
    return init;
}

MockSampleBox::MockSampleBox(ArrayBuffer* data)
    : MockBox(data)
{
    ASSERT(m_length == 30);

    auto view = JSC::DataView::create(data, 0, data->byteLength());
    int32_t timeScale = view->get<int32_t>(8, true);

    int32_t timeValue = view->get<int32_t>(12, true);
    m_presentationTimestamp = MediaTime(timeValue, timeScale);

    timeValue = view->get<int32_t>(16, true);
    m_decodeTimestamp = MediaTime(timeValue, timeScale);

    timeValue = view->get<int32_t>(20, true);
    m_duration = MediaTime(timeValue, timeScale);

    m_trackID = view->get<int32_t>(24, true);
    m_flags = view->get<uint8_t>(28, true);
    m_generation = view->get<uint8_t>(29, true);
}

const String& MockSampleBox::type()
{
    static NeverDestroyed<String> smpl(MAKE_STATIC_STRING_IMPL("smpl"));
    return smpl;
}

}

#endif
