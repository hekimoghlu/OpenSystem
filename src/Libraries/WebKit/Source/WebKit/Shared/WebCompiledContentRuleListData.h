/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include <WebCore/SharedBuffer.h>
#include <WebCore/SharedMemory.h>
#include <variant>
#include <wtf/RefPtr.h>

namespace WebKit {

class WebCompiledContentRuleListData {
public:
    WebCompiledContentRuleListData(String&& identifier, Ref<WebCore::SharedMemory>&& data, size_t actionsOffset, size_t actionsSize, size_t urlFiltersBytecodeOffset, size_t urlFiltersBytecodeSize, size_t topURLFiltersBytecodeOffset, size_t topURLFiltersBytecodeSize, size_t frameURLFiltersBytecodeOffset, size_t frameURLFiltersBytecodeSize)
        : identifier(WTFMove(identifier))
        , data(WTFMove(data))
        , actionsOffset(actionsOffset)
        , actionsSize(actionsSize)
        , urlFiltersBytecodeOffset(urlFiltersBytecodeOffset)
        , urlFiltersBytecodeSize(urlFiltersBytecodeSize)
        , topURLFiltersBytecodeOffset(topURLFiltersBytecodeOffset)
        , topURLFiltersBytecodeSize(topURLFiltersBytecodeSize)
        , frameURLFiltersBytecodeOffset(frameURLFiltersBytecodeOffset)
        , frameURLFiltersBytecodeSize(frameURLFiltersBytecodeSize)
    {
    }

    WebCompiledContentRuleListData(String&& identifier, std::optional<WebCore::SharedMemoryHandle>&& data, size_t actionsOffset, size_t actionsSize, size_t urlFiltersBytecodeOffset, size_t urlFiltersBytecodeSize, size_t topURLFiltersBytecodeOffset, size_t topURLFiltersBytecodeSize, size_t frameURLFiltersBytecodeOffset, size_t frameURLFiltersBytecodeSize);

    std::optional<WebCore::SharedMemoryHandle> createDataHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadOnly) const;

    String identifier;
    RefPtr<WebCore::SharedMemory> data;
    size_t actionsOffset { 0 };
    size_t actionsSize { 0 };
    size_t urlFiltersBytecodeOffset { 0 };
    size_t urlFiltersBytecodeSize { 0 };
    size_t topURLFiltersBytecodeOffset { 0 };
    size_t topURLFiltersBytecodeSize { 0 };
    size_t frameURLFiltersBytecodeOffset { 0 };
    size_t frameURLFiltersBytecodeSize { 0 };
};

}

#endif // ENABLE(CONTENT_EXTENSIONS)
