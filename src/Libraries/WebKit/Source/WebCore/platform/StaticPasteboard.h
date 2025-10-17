/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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

#include "Pasteboard.h"
#include <wtf/RobinHoodHashSet.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class SharedBuffer;

class StaticPasteboard final : public Pasteboard {
public:
    StaticPasteboard();
    ~StaticPasteboard();

    PasteboardCustomData takeCustomData();

    bool hasNonDefaultData() const;
    bool isStatic() const final { return true; }

    bool hasData() final;
    Vector<String> typesSafeForBindings(const String&) final;
    Vector<String> typesForLegacyUnsafeBindings() final;
    String readOrigin() final { return { }; }
    String readString(const String& type) final;
    String readStringInCustomData(const String& type) final;

    void writeString(const String& type, const String& data) final;
    void writeData(const String& type, Ref<SharedBuffer>&& data);
    void writeStringInCustomData(const String& type, const String& data);
    void clear() final;
    void clear(const String& type) final;

    void read(PasteboardPlainText&, PlainTextURLReadingPolicy = PlainTextURLReadingPolicy::AllowURL, std::optional<size_t> = std::nullopt) final { }
    void read(PasteboardWebContentReader&, WebContentReadingPolicy, std::optional<size_t> = std::nullopt) final { }

    void write(const PasteboardURL&) final;
    void write(const PasteboardImage&) final;
    void write(const PasteboardWebContent&) final;
    void writeMarkup(const String&) final;
    void writePlainText(const String&, SmartReplaceOption) final;

    void writeCustomData(const Vector<PasteboardCustomData>&) final { }

    Pasteboard::FileContentState fileContentState() final { return m_fileContentState; }
    bool canSmartReplace() final { return false; }

#if ENABLE(DRAG_SUPPORT)
    void setDragImage(DragImage, const IntPoint&) final { }
#endif

private:
    PasteboardCustomData m_customData;
    MemoryCompactRobinHoodHashSet<String> m_nonDefaultDataTypes;
    Pasteboard::FileContentState m_fileContentState { Pasteboard::FileContentState::NoFileOrImageData };
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StaticPasteboard)
    static bool isType(const WebCore::Pasteboard& pasteboard) { return pasteboard.isStatic(); }
SPECIALIZE_TYPE_TRAITS_END()
