/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#include <WebCore/SharedStringHash.h>
#include <wtf/RefPtr.h>

namespace WebCore {
class SharedMemory;
}

namespace WebKit {

class SharedStringHashTableReadOnly {
public:
    SharedStringHashTableReadOnly();
    ~SharedStringHashTableReadOnly();

    bool contains(WebCore::SharedStringHash) const;

    WebCore::SharedMemory* sharedMemory() const { return m_sharedMemory.get(); }
    void setSharedMemory(RefPtr<WebCore::SharedMemory>&&);

protected:
    WebCore::SharedStringHash* findSlot(WebCore::SharedStringHash) const;

    RefPtr<WebCore::SharedMemory> m_sharedMemory;
    unsigned m_tableSizeMask { 0 };
    std::span<WebCore::SharedStringHash> m_table;
};

} // namespace WebKit
