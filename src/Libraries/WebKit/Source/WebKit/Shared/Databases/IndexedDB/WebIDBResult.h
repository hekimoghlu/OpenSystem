/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

#include "SandboxExtension.h"
#include <WebCore/IDBResultData.h>
#include <wtf/Noncopyable.h>

namespace WebKit {

class WebIDBResult {
    WTF_MAKE_NONCOPYABLE(WebIDBResult);
public:
    WebIDBResult()
    {
    }

    WebIDBResult(const WebCore::IDBResultData& resultData)
        : m_resultData(resultData)
    {
    }

    WebIDBResult(const WebCore::IDBResultData& resultData, Vector<SandboxExtension::Handle>&& handles)
        : m_resultData(resultData)
        , m_handles(WTFMove(handles))
    {
    }
    
    WebIDBResult(WebIDBResult&&) = default;
    WebIDBResult& operator=(WebIDBResult&&) = default;

    const WebCore::IDBResultData& resultData() const { return m_resultData; }
    const Vector<SandboxExtension::Handle>& handles() const { return m_handles; }

private:
    friend struct IPC::ArgumentCoder<WebIDBResult, void>;
    WebCore::IDBResultData m_resultData;
    Vector<SandboxExtension::Handle> m_handles;
};

} // namespace WebKit
