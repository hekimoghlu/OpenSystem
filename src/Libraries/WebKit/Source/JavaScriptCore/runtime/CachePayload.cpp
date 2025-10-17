/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#include "CachePayload.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

CachePayload CachePayload::makeMappedPayload(FileSystem::MappedFileData&& data)
{
    return CachePayload(WTFMove(data));
}

CachePayload CachePayload::makeMallocPayload(MallocSpan<uint8_t, VMMalloc>&& data)
{
    return CachePayload(WTFMove(data));
}

CachePayload CachePayload::makeEmptyPayload()
{
    return CachePayload({ });
}

CachePayload::CachePayload(CachePayload&&) = default;

CachePayload::CachePayload(DataType&& data)
    : m_data(WTFMove(data))
{
}

CachePayload::~CachePayload() = default;

std::span<const uint8_t> CachePayload::span() const
{
    return WTF::switchOn(m_data, [](const auto& data) {
        return data.span();
    });
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
