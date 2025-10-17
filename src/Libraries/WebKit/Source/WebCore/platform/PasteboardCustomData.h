/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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

#include <variant>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/Vector.h>
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class SharedBuffer;

class PasteboardCustomData {
public:
    struct Entry {
        WEBCORE_EXPORT Entry();
        WEBCORE_EXPORT Entry(const Entry&);
        WEBCORE_EXPORT Entry(const String&, const String&, const std::variant<String, Ref<WebCore::SharedBuffer>>&);
        WEBCORE_EXPORT Entry(Entry&&);
        WEBCORE_EXPORT Entry& operator=(const Entry& otherData);
        WEBCORE_EXPORT Entry& operator=(Entry&& otherData);
        Entry(const String& type);

        String type;
        String customData;
        std::variant<String, Ref<SharedBuffer>> platformData;
    };

    WEBCORE_EXPORT PasteboardCustomData();
    WEBCORE_EXPORT PasteboardCustomData(String&& origin, Vector<Entry>&&);
    WEBCORE_EXPORT PasteboardCustomData(PasteboardCustomData&&);
    WEBCORE_EXPORT PasteboardCustomData(const PasteboardCustomData&);
    WEBCORE_EXPORT ~PasteboardCustomData();

    const String& origin() const { return m_origin; }
    void setOrigin(const String& origin) { m_origin = origin; }

    WEBCORE_EXPORT Ref<SharedBuffer> createSharedBuffer() const;
    WEBCORE_EXPORT static PasteboardCustomData fromSharedBuffer(const SharedBuffer&);
    WEBCORE_EXPORT static PasteboardCustomData fromPersistenceDecoder(WTF::Persistence::Decoder&&);

    String readString(const String& type) const;
    RefPtr<SharedBuffer> readBuffer(const String& type) const;
    String readStringInCustomData(const String& type) const;

    void writeString(const String& type, const String& value);
    void writeData(const String& type, Ref<SharedBuffer>&& data);
    void writeStringInCustomData(const String& type, const String& value);

    void clear();
    void clear(const String& type);

#if PLATFORM(COCOA)
    WEBCORE_EXPORT static ASCIILiteral cocoaType();
#elif PLATFORM(GTK)
    static ASCIILiteral gtkType() { return "org.webkitgtk.WebKit.custom-pasteboard-data"_s; }
#endif

    void forEachType(Function<void(const String&)>&&) const;
    void forEachPlatformString(Function<void(const String& type, const String& data)>&&) const;
    void forEachPlatformStringOrBuffer(Function<void(const String& type, const std::variant<String, Ref<SharedBuffer>>& data)>&&) const;
    void forEachCustomString(Function<void(const String& type, const String& data)>&&) const;

    bool hasData() const;
    bool hasSameOriginCustomData() const;

    Vector<String> orderedTypes() const;
    WEBCORE_EXPORT PasteboardCustomData& operator=(const PasteboardCustomData& otherData);

    const Vector<Entry>& data() const { return m_data; }

private:
    UncheckedKeyHashMap<String, String> sameOriginCustomStringData() const;
    Entry& addOrMoveEntryToEnd(const String&);

    String m_origin;
    Vector<Entry> m_data;
};

} // namespace WebCore
