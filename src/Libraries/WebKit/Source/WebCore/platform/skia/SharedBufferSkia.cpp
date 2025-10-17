/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
#include "SharedBuffer.h"

namespace WebCore {

FragmentedSharedBuffer::FragmentedSharedBuffer(sk_sp<SkData>&& data)
{
    ASSERT(data);
    m_size = data->size();
    m_segments.append({ 0, DataSegment::create(WTFMove(data)) });
}

Ref<FragmentedSharedBuffer> FragmentedSharedBuffer::create(sk_sp<SkData>&& data)
{
    return adoptRef(*new FragmentedSharedBuffer(WTFMove(data)));
}

sk_sp<SkData> SharedBuffer::createSkData() const
{
    ref();
    return SkData::MakeWithProc(span().data(), size(), [](const void*, void* context) {
        static_cast<SharedBuffer*>(context)->deref();
    }, const_cast<SharedBuffer*>(this));
}

} // namespace WebCore
