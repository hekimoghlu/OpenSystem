/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#include "WebCompiledContentRuleList.h"

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebKit {

RefPtr<WebCompiledContentRuleList> WebCompiledContentRuleList::create(WebCompiledContentRuleListData&& data)
{
    if (!data.data)
        return nullptr;
    return adoptRef(*new WebCompiledContentRuleList(WTFMove(data)));
}

WebCompiledContentRuleList::WebCompiledContentRuleList(WebCompiledContentRuleListData&& data)
    : m_data(WTFMove(data))
{
}

WebCompiledContentRuleList::~WebCompiledContentRuleList()
{
}

std::span<const uint8_t> WebCompiledContentRuleList::urlFiltersBytecode() const
{
    return spanWithOffsetAndLength(m_data.urlFiltersBytecodeOffset, m_data.urlFiltersBytecodeSize);
}

std::span<const uint8_t> WebCompiledContentRuleList::topURLFiltersBytecode() const
{
    return spanWithOffsetAndLength(m_data.topURLFiltersBytecodeOffset, m_data.topURLFiltersBytecodeSize);
}

std::span<const uint8_t> WebCompiledContentRuleList::frameURLFiltersBytecode() const
{
    return spanWithOffsetAndLength(m_data.frameURLFiltersBytecodeOffset, m_data.frameURLFiltersBytecodeSize);
}

std::span<const uint8_t> WebCompiledContentRuleList::serializedActions() const
{
    return spanWithOffsetAndLength(m_data.actionsOffset, m_data.actionsSize);
}

std::span<const uint8_t> WebCompiledContentRuleList::spanWithOffsetAndLength(size_t offset, size_t length) const
{
    RELEASE_ASSERT(offset + length <= m_data.data->size());
    return m_data.data->span().subspan(offset, length);
}

} // namespace WebKit

#endif // ENABLE(CONTENT_EXTENSIONS)
