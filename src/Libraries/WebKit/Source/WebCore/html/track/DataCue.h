/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#if ENABLE(VIDEO)

#include "SerializedPlatformDataCue.h"
#include "TextTrackCue.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/MediaTime.h>
#include <wtf/TypeCasts.h>

namespace JSC {
class ArrayBuffer;
class JSValue;
}

namespace WebCore {

class ScriptExecutionContext;

class DataCue final : public TextTrackCue {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DataCue);
public:
    static Ref<DataCue> create(Document&, double start, double end, ArrayBuffer& data);
    static Ref<DataCue> create(Document&, double start, double end, JSC::JSValue, const String& type);
    static Ref<DataCue> create(Document&, const MediaTime& start, const MediaTime& end, std::span<const uint8_t> data);
    static Ref<DataCue> create(Document&, const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&&, const String& type);

    virtual ~DataCue();

    RefPtr<JSC::ArrayBuffer> data() const;
    void setData(JSC::ArrayBuffer&);

    const SerializedPlatformDataCue* platformValue() const { return m_platformValue.get(); }

    JSC::JSValue value(JSC::JSGlobalObject&) const;
    void setValue(JSC::JSGlobalObject&, JSC::JSValue);

    String type() const { return m_type; }
    void setType(const String& type) { m_type = type; }

private:
    DataCue(Document&, const MediaTime& start, const MediaTime& end, ArrayBuffer&, const String&);
    DataCue(Document&, const MediaTime& start, const MediaTime& end, std::span<const uint8_t>);
    DataCue(Document&, const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&&, const String&);
    DataCue(Document&, const MediaTime& start, const MediaTime& end, JSC::JSValue, const String&);

    JSC::JSValue valueOrNull() const;
    CueType cueType() const final { return Data; }
    bool cueContentsMatch(const TextTrackCue&) const final;
    void toJSON(JSON::Object&) const final;

    RefPtr<ArrayBuffer> m_data;
    String m_type;
    RefPtr<SerializedPlatformDataCue> m_platformValue;
    // FIXME: The following use of JSC::Strong is incorrect and can lead to storage leaks
    // due to reference cycles; we should use JSValueInWrappedObject instead.
    // https://bugs.webkit.org/show_bug.cgi?id=201173
    JSC::Strong<JSC::Unknown> m_value;
};

} // namespace WebCore

namespace WTF {

template<> struct LogArgument<WebCore::DataCue> : LogArgument<WebCore::TextTrackCue> { };

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::DataCue)
static bool isType(const WebCore::TextTrackCue& cue) { return cue.cueType() == WebCore::TextTrackCue::Data; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
