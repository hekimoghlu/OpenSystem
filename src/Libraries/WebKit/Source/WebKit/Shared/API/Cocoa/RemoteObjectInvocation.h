/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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

#include "APIDictionary.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace IPC {
class Encoder;
class Decoder;
}

namespace WebKit {

class RemoteObjectInvocation {
    WTF_MAKE_TZONE_ALLOCATED(RemoteObjectInvocation);
public:
    struct ReplyInfo {
        WTF_MAKE_STRUCT_TZONE_ALLOCATED(ReplyInfo);

        ReplyInfo(uint64_t replyID, String&& blockSignature)
            : replyID(replyID)
            , blockSignature(WTFMove(blockSignature))
        {
        }

        const uint64_t replyID;
        const String blockSignature;
    };
    RemoteObjectInvocation();
    RemoteObjectInvocation(const String& interfaceIdentifier, RefPtr<API::Dictionary>&& encodedInvocation, std::unique_ptr<ReplyInfo>&&);

    const String& interfaceIdentifier() const { return m_interfaceIdentifier; }
    const RefPtr<API::Dictionary>& encodedInvocation() const { return m_encodedInvocation; }
    const std::unique_ptr<ReplyInfo>& replyInfo() const { return m_replyInfo; }

private:
    String m_interfaceIdentifier;
    RefPtr<API::Dictionary> m_encodedInvocation;
    std::unique_ptr<ReplyInfo> m_replyInfo;
};

}
