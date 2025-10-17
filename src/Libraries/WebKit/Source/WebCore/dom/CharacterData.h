/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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

#include "ContainerNode.h"

namespace WebCore {

class CharacterData : public Node {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CharacterData);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(CharacterData);
public:
    const String& data() const { return m_data; }
    static constexpr ptrdiff_t dataMemoryOffset() { return OBJECT_OFFSETOF(CharacterData, m_data); }

    WEBCORE_EXPORT void setData(const String&);
    unsigned length() const { return m_data.length(); }
    WEBCORE_EXPORT ExceptionOr<String> substringData(unsigned offset, unsigned count) const;
    WEBCORE_EXPORT void appendData(const String&);
    WEBCORE_EXPORT ExceptionOr<void> insertData(unsigned offset, const String&);
    WEBCORE_EXPORT ExceptionOr<void> deleteData(unsigned offset, unsigned count);
    WEBCORE_EXPORT ExceptionOr<void> replaceData(unsigned offset, unsigned count, const String&);

    bool containsOnlyASCIIWhitespace() const;

    // Like appendData, but optimized for the parser (e.g., no mutation events).
    void parserAppendData(StringView);

protected:
    CharacterData(Document& document, String&& text, NodeType type, OptionSet<TypeFlag> typeFlags = { })
        : Node(document, type, typeFlags | TypeFlag::IsCharacterData)
        , m_data(!text.isNull() ? WTFMove(text) : emptyString())
    {
        ASSERT(isCharacterDataNode());
        ASSERT(!isContainerNode());
    }

    ~CharacterData();

    void setDataWithoutUpdate(const String&);

    void dispatchModifiedEvent(const String& oldValue);

    enum class UpdateLiveRanges : bool { No, Yes };
    virtual void setDataAndUpdate(const String&, unsigned offsetOfReplacedData, unsigned oldLength, unsigned newLength, UpdateLiveRanges = UpdateLiveRanges::Yes);

private:
    String nodeValue() const final;
    ExceptionOr<void> setNodeValue(const String&) final;
    void notifyParentAfterChange(const ContainerNode::ChildChange&);

    void parentOrShadowHostNode() const = delete; // Call parentNode() instead.

    String m_data;
};

inline unsigned Node::length() const
{
    if (auto characterData = dynamicDowncast<CharacterData>(*this))
        return characterData->length();
    return countChildNodes();
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CharacterData)
    static bool isType(const WebCore::Node& node) { return node.isCharacterDataNode(); }
SPECIALIZE_TYPE_TRAITS_END()
