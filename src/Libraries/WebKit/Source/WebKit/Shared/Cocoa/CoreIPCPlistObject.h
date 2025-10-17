/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

#if PLATFORM(COCOA)

#include "ArgumentCodersCocoa.h"
#include <wtf/RetainPtr.h>
#include <wtf/UniqueRef.h>

namespace WebKit {

class CoreIPCPlistArray;
class CoreIPCPlistDictionary;
class CoreIPCString;
class CoreIPCNumber;
class CoreIPCDate;
class CoreIPCData;

using PlistValue = std::variant<
    CoreIPCPlistArray,
    CoreIPCPlistDictionary,
    CoreIPCString,
    CoreIPCNumber,
    CoreIPCDate,
    CoreIPCData
>;

class CoreIPCPlistObject {
public:
    CoreIPCPlistObject(id);
    CoreIPCPlistObject(UniqueRef<PlistValue>&&);

    RetainPtr<id> toID() const;
    static bool isPlistType(id);

    const UniqueRef<PlistValue>& value() const { return m_value; }
private:
    UniqueRef<PlistValue> m_value;
};

} // namespace WebKit

namespace IPC {

// This ArgumentCoders specialization for UniqueRef<PlistValue> is to allow us to use
// makeUniqueRefWithoutFastMallocCheck<>, since we can't make the variant fast malloc'ed
template<> struct ArgumentCoder<UniqueRef<WebKit::PlistValue>> {
    static void encode(Encoder&, const UniqueRef<WebKit::PlistValue>&);
    static std::optional<UniqueRef<WebKit::PlistValue>> decode(Decoder&);
};

} // namespace IPC

#endif // PLATFORM(COCOA)
