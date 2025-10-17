/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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

#include <wtf/ArgumentCoder.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

typedef struct _GUnixFDList GUnixFDList;
typedef struct _GVariant GVariant;

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

struct UserMessage {
    enum class Type : uint8_t {
        Null,
        Message,
        Error,
    };

    UserMessage()
        : type(Type::Null)
    {
    }

    UserMessage(const CString& name, uint32_t errorCode)
        : type(Type::Error)
        , name(name)
        , errorCode(errorCode)
    {
    }

    UserMessage(const CString& name, GRefPtr<GVariant>& parameters, GRefPtr<GUnixFDList>& fileDescriptors)
        : type(Type::Message)
        , name(name)
        , parameters(parameters)
        , fileDescriptors(fileDescriptors)
    {
    }

    Type type { Type::Null };
    CString name;
    GRefPtr<GVariant> parameters;
    GRefPtr<GUnixFDList> fileDescriptors;
    uint32_t errorCode { 0 };

    struct NullMessage {
    };

    struct ErrorMessage {
        CString name;
        uint32_t errorCode;
    };

    struct DataMessage {
        CString name;
        GRefPtr<GVariant> parameters;
        GRefPtr<GUnixFDList> fileDescriptors;
    };

private:
    friend struct IPC::ArgumentCoder<UserMessage, void>;

    using IPCData = std::variant<NullMessage, ErrorMessage, DataMessage>;
    static UserMessage fromIPCData(IPCData&&);
    IPCData toIPCData() const;
};

} // namespace WebKit
