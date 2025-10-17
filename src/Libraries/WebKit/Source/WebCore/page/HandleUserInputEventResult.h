/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "RemoteUserInputEventData.h"

namespace WebCore {

class HandleUserInputEventResult {
public:
    HandleUserInputEventResult(bool handled)
        : m_result(handled)
    {
    }

    HandleUserInputEventResult(RemoteUserInputEventData remoteUserInputEventData)
        : m_result(makeUnexpected(remoteUserInputEventData))
    {
    }

    bool wasHandled() { return m_result ? *m_result : false; }
    void setHandled(bool handled)
    {
        if (m_result.has_value())
            m_result = handled;
    }

    std::optional<RemoteUserInputEventData> remoteUserInputEventData()
    {
        return m_result ? std::nullopt : std::optional<RemoteUserInputEventData>(m_result.error());
    }
private:
    Expected<bool, RemoteUserInputEventData> m_result;
};

}
