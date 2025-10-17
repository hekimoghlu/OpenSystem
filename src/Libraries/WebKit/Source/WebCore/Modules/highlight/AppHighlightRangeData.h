/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

#include <wtf/Forward.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

#if ENABLE(APP_HIGHLIGHTS)

class FragmentedSharedBuffer;

class AppHighlightRangeData {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(AppHighlightRangeData, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static std::optional<AppHighlightRangeData> create(const FragmentedSharedBuffer&);
    struct NodePathComponent {
        String identifier;
        String nodeName;
        String textData;
        uint32_t pathIndex { 0 };

        NodePathComponent(String&& elementIdentifier, String&& name, String&& data, uint32_t index)
            : identifier(WTFMove(elementIdentifier))
            , nodeName(WTFMove(name))
            , textData(WTFMove(data))
            , pathIndex(index)
        {
        }

        NodePathComponent(const String& elementIdentifier, const String& name, const String& data, uint32_t index)
            : identifier(elementIdentifier)
            , nodeName(name)
            , textData(data)
            , pathIndex(index)
        {
        }

        friend bool operator==(const NodePathComponent&, const NodePathComponent&) = default;
    };

    using NodePath = Vector<NodePathComponent>;

    AppHighlightRangeData(const AppHighlightRangeData&) = default;
    AppHighlightRangeData() = default;
    AppHighlightRangeData(String&& identifier, String&& text, NodePath&& startContainer, uint64_t startOffset, NodePath&& endContainer, uint64_t endOffset)
        : m_identifier(WTFMove(identifier))
        , m_text(WTFMove(text))
        , m_startContainer(WTFMove(startContainer))
        , m_startOffset(startOffset)
        , m_endContainer(WTFMove(endContainer))
        , m_endOffset(endOffset)
    {
    }

    AppHighlightRangeData(const String& identifier, const String& text, const NodePath& startContainer, uint64_t startOffset, const NodePath& endContainer, uint64_t endOffset)
        : m_identifier(identifier)
        , m_text(text)
        , m_startContainer(startContainer)
        , m_startOffset(startOffset)
        , m_endContainer(endContainer)
        , m_endOffset(endOffset)
    {
    }

    AppHighlightRangeData& operator=(const AppHighlightRangeData&) = default;

    const String& identifier() const { return m_identifier; }
    const String& text() const { return m_text; }
    const NodePath& startContainer() const { return m_startContainer; }
    uint32_t startOffset() const { return m_startOffset; }
    const NodePath& endContainer() const { return m_endContainer; }
    uint32_t endOffset() const { return m_endOffset; }

    Ref<FragmentedSharedBuffer> toSharedBuffer() const;

private:
    String m_identifier;
    String m_text;
    NodePath m_startContainer;
    uint32_t m_startOffset { 0 };
    NodePath m_endContainer;
    uint32_t m_endOffset { 0 };
};

#endif

} // namespace WebCore
