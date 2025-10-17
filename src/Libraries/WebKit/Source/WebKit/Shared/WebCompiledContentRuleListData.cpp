/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#include "WebCompiledContentRuleListData.h"

#if ENABLE(CONTENT_EXTENSIONS)

#include "ArgumentCoders.h"

namespace WebKit {

static size_t ruleListDataSize(size_t topURLFiltersBytecodeOffset, size_t topURLFiltersBytecodeSize)
{
    return topURLFiltersBytecodeOffset + topURLFiltersBytecodeSize;
}

std::optional<WebCore::SharedMemoryHandle> WebCompiledContentRuleListData::createDataHandle(WebCore::SharedMemory::Protection protection) const
{
    return data->createHandle(protection);
}

WebCompiledContentRuleListData::WebCompiledContentRuleListData(String&& identifier, std::optional<WebCore::SharedMemoryHandle>&& dataHandle, size_t actionsOffset, size_t actionsSize, size_t urlFiltersBytecodeOffset, size_t urlFiltersBytecodeSize, size_t topURLFiltersBytecodeOffset, size_t topURLFiltersBytecodeSize, size_t frameURLFiltersBytecodeOffset, size_t frameURLFiltersBytecodeSize)
    : identifier(WTFMove(identifier))
    , data(dataHandle ? WebCore::SharedMemory::map(WTFMove(*dataHandle), WebCore::SharedMemory::Protection::ReadOnly) : nullptr)
    , actionsOffset(actionsOffset)
    , actionsSize(actionsSize)
    , urlFiltersBytecodeOffset(urlFiltersBytecodeOffset)
    , urlFiltersBytecodeSize(urlFiltersBytecodeSize)
    , topURLFiltersBytecodeOffset(topURLFiltersBytecodeOffset)
    , topURLFiltersBytecodeSize(topURLFiltersBytecodeSize)
    , frameURLFiltersBytecodeOffset(frameURLFiltersBytecodeOffset)
    , frameURLFiltersBytecodeSize(frameURLFiltersBytecodeSize)
{
    if (data) {
        if (data->size() < ruleListDataSize(actionsOffset, actionsSize)
        || data->size() < ruleListDataSize(urlFiltersBytecodeOffset, urlFiltersBytecodeSize)
        || data->size() < ruleListDataSize(topURLFiltersBytecodeOffset, topURLFiltersBytecodeSize)
        || data->size() < ruleListDataSize(frameURLFiltersBytecodeOffset, frameURLFiltersBytecodeSize)) {
            ASSERT_NOT_REACHED();
            data = nullptr;
        }
    }
}

} // namespace WebKit

#endif // ENABLE(CONTENT_EXTENSIONS)
