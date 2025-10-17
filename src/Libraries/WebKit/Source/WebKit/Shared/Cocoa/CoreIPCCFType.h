/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

#include "CoreIPCRetainPtr.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/UniqueRef.h>

namespace WebCore {
class Color;
}

namespace WebKit {

class CoreIPCCFArray;
class CoreIPCBoolean;
class CoreIPCCFCharacterSet;
class CoreIPCColor;
class CoreIPCData;
class CoreIPCDate;
class CoreIPCCFDictionary;
class CoreIPCNull;
class CoreIPCNumber;
class CoreIPCString;
class CoreIPCCFURL;
class CoreIPCCGColorSpace;
class CoreIPCSecCertificate;
class CoreIPCSecTrust;
#if HAVE(SEC_ACCESS_CONTROL)
class CoreIPCSecAccessControl;
#endif

using CFObjectValue = std::variant<
    std::nullptr_t,
    CoreIPCCFArray,
    CoreIPCBoolean,
    CoreIPCCFCharacterSet,
    CoreIPCData,
    CoreIPCDate,
    CoreIPCCFDictionary,
    CoreIPCNull,
    CoreIPCNumber,
    CoreIPCString,
    CoreIPCCFURL,
    CoreIPCSecCertificate,
    CoreIPCSecTrust,
    CoreIPCCGColorSpace,
    WebCore::Color
#if HAVE(SEC_ACCESS_CONTROL)
    , CoreIPCSecAccessControl
#endif
>;

class CoreIPCCFType {
public:
    CoreIPCCFType(CFTypeRef);
    CoreIPCCFType(CoreIPCCFType&&);
    CoreIPCCFType(UniqueRef<CFObjectValue>&&);
    ~CoreIPCCFType();

    const UniqueRef<CFObjectValue>& object() const { return m_object; }
    RetainPtr<id> toID() const;
    RetainPtr<CFTypeRef> toCFType() const;

private:
    UniqueRef<CFObjectValue> m_object;
};

} // namespace WebKit

namespace IPC {

// This ArgumentCoders specialization for UniqueRef<CFObjectValue> is to allow us to use
// makeUniqueRefWithoutFastMallocCheck<>, since we can't make the variant fast malloc'ed
template<> struct ArgumentCoder<UniqueRef<WebKit::CFObjectValue>> {
    template<typename Encoder>
    static void encode(Encoder&, const UniqueRef<WebKit::CFObjectValue>&);
    static std::optional<UniqueRef<WebKit::CFObjectValue>> decode(Decoder&);
};

} // namespace IPC
