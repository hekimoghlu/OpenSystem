/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

#if ENABLE(VIDEO)

#include "DataCue.h"

#include "Document.h"
#include "Logging.h"
#include "TextTrack.h"
#include "TextTrackCueList.h"
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/StrongInlines.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DataCue);

DataCue::DataCue(Document& document, const MediaTime& start, const MediaTime& end, ArrayBuffer& data, const String& type)
    : TextTrackCue(document, start, end)
    , m_type(type)
{
    setData(data);
}

DataCue::DataCue(Document& document, const MediaTime& start, const MediaTime& end, std::span<const uint8_t> data)
    : TextTrackCue(document, start, end)
    , m_data(ArrayBuffer::create(data))
{
}

DataCue::DataCue(Document& document, const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&& platformValue, const String& type)
    : TextTrackCue(document, start, end)
    , m_type(type)
    , m_platformValue(WTFMove(platformValue))
{
}

DataCue::DataCue(Document& document, const MediaTime& start, const MediaTime& end, JSC::JSValue value, const String& type)
    : TextTrackCue(document, start, end)
    , m_type(type)
    , m_value(document.vm(), value)
{
}

Ref<DataCue> DataCue::create(Document& document, const MediaTime& start, const MediaTime& end, std::span<const uint8_t> data)
{
    auto dataCue = adoptRef(*new DataCue(document, start, end, data));
    dataCue->suspendIfNeeded();
    return dataCue;
}

Ref<DataCue> DataCue::create(Document& document, const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&& platformValue, const String& type)
{
    auto dataCue = adoptRef(*new DataCue(document, start, end, WTFMove(platformValue), type));
    dataCue->suspendIfNeeded();
    return dataCue;
}

Ref<DataCue> DataCue::create(Document& document, double start, double end, ArrayBuffer& data)
{
    auto dataCue = adoptRef(*new DataCue(document, MediaTime::createWithDouble(start), MediaTime::createWithDouble(end), data, emptyString()));
    dataCue->suspendIfNeeded();
    return dataCue;
}

Ref<DataCue> DataCue::create(Document& document, double start, double end, JSC::JSValue value, const String& type)
{
    auto dataCue = adoptRef(*new DataCue(document, MediaTime::createWithDouble(start), MediaTime::createWithDouble(end), value, type));
    dataCue->suspendIfNeeded();
    return dataCue;
}

DataCue::~DataCue() = default;

RefPtr<ArrayBuffer> DataCue::data() const
{
    if (m_platformValue)
        return m_platformValue->data();

    if (!m_data)
        return nullptr;

    return ArrayBuffer::create(*m_data);
}

void DataCue::setData(ArrayBuffer& data)
{
    m_platformValue = nullptr;
    m_value.clear();
    m_data = ArrayBuffer::create(data);
}

bool DataCue::cueContentsMatch(const TextTrackCue& cue) const
{
    const DataCue* dataCue = downcast<DataCue>(&cue);
    RefPtr<ArrayBuffer> otherData = dataCue->data();
    if ((otherData && !m_data) || (!otherData && m_data))
        return false;
    if (m_data && m_data->data() && !equalSpans(m_data->span(), otherData->span()))
        return false;

    auto otherPlatformValue = dataCue->platformValue();
    if ((otherPlatformValue && !m_platformValue) || (!otherPlatformValue && m_platformValue))
        return false;
    if (m_platformValue && !m_platformValue->isEqual(*otherPlatformValue))
        return false;

    JSC::JSValue thisValue = valueOrNull();
    JSC::JSValue otherValue = dataCue->valueOrNull();
    if ((otherValue && !thisValue) || (!otherValue && thisValue))
        return false;
    if (!JSC::JSValue::strictEqual(nullptr, thisValue, otherValue))
        return false;

    return true;
}

JSC::JSValue DataCue::value(JSC::JSGlobalObject& state) const
{
    if (m_platformValue)
        return m_platformValue->deserialize(&state);

    if (m_value)
        return m_value.get();

    return JSC::jsNull();
}

void DataCue::setValue(JSC::JSGlobalObject& state, JSC::JSValue value)
{
    // FIXME: this should use a SerializedScriptValue.
    m_value.set(state.vm(), value);
    m_platformValue = nullptr;
    m_data = nullptr;
}

JSValue DataCue::valueOrNull() const
{
    if (m_value)
        return m_value.get();

    return jsNull();
}

void DataCue::toJSON(JSON::Object& object) const
{
    TextTrackCue::toJSON(object);

    if (!m_type.isEmpty())
        object.setString("type"_s, m_type);
}

} // namespace WebCore

#endif
