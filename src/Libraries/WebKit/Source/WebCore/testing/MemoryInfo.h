/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

#include "CommonVM.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class MemoryInfo : public RefCounted<MemoryInfo> {
public:
    static Ref<MemoryInfo> create() { return adoptRef(*new MemoryInfo); }

    size_t usedJSHeapSize() const { return m_usedJSHeapSize; }
    size_t totalJSHeapSize() const { return m_totalJSHeapSize; }

private:
    MemoryInfo()
        : m_usedJSHeapSize(commonVM().heap.size())
        , m_totalJSHeapSize(commonVM().heap.capacity())
    {
    }

    size_t m_usedJSHeapSize;
    size_t m_totalJSHeapSize;
};

} // namespace WebCore
