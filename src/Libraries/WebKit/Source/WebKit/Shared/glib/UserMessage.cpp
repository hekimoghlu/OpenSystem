/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#include "UserMessage.h"

#include "ArgumentCodersGLib.h"

namespace WebKit {

UserMessage::IPCData UserMessage::toIPCData() const
{
    switch (type) {
    case Type::Null:
        return NullMessage { };
    case Type::Error:
        return ErrorMessage { name, errorCode };
    case Type::Message:
        return DataMessage { name, parameters, fileDescriptors };
    }

    ASSERT_NOT_REACHED();

    return NullMessage { };
}

UserMessage UserMessage::fromIPCData(UserMessage::IPCData&& ipcData)
{
    return WTF::switchOn(WTFMove(ipcData),
        [&] (NullMessage&&) {
            return UserMessage { };
        },
        [&] (ErrorMessage&& message) {
            return UserMessage { message.name, message.errorCode };
        },
        [&] (DataMessage&& message) {
            return UserMessage { message.name, message.parameters, message.fileDescriptors };
        }
    );
}

}
