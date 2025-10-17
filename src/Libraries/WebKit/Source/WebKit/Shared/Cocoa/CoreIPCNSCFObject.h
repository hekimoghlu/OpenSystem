/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

class CoreIPCArray;
class CoreIPCCFType;
class CoreIPCColor;
#if USE(PASSKIT)
class CoreIPCPKPaymentMethod;
class CoreIPCPKPaymentMerchantSession;
class CoreIPCPKPaymentSetupFeature;
class CoreIPCPKContact;
class CoreIPCPKSecureElementPass;
class CoreIPCPKPayment;
class CoreIPCPKPaymentToken;
class CoreIPCPKShippingMethod;
class CoreIPCPKDateComponentsRange;
class CoreIPCCNContact;
class CoreIPCCNPhoneNumber;
class CoreIPCCNPostalAddress;
#endif
#if ENABLE(DATA_DETECTION) && HAVE(WK_SECURE_CODING_DATA_DETECTORS)
class CoreIPCDDScannerResult;
#if PLATFORM(MAC)
class CoreIPCDDSecureActionContext;
#endif
#endif
class CoreIPCData;
class CoreIPCDate;
class CoreIPCDateComponents;
class CoreIPCDictionary;
class CoreIPCError;
class CoreIPCFont;
class CoreIPCLocale;
class CoreIPCNSShadow;
class CoreIPCNSValue;
class CoreIPCNumber;
class CoreIPCNull;
#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
class CoreIPCSecureCoding;
#endif
class CoreIPCString;
class CoreIPCURL;

using ObjectValue = std::variant<
    std::nullptr_t,
    CoreIPCArray,
    CoreIPCCFType,
    CoreIPCColor,
    CoreIPCData,
    CoreIPCDate,
    CoreIPCDictionary,
    CoreIPCError,
    CoreIPCFont,
    CoreIPCLocale,
    CoreIPCNSShadow,
    CoreIPCNSValue,
    CoreIPCNumber,
    CoreIPCNull,
#if USE(PASSKIT)
    CoreIPCPKPaymentMethod,
    CoreIPCPKPaymentMerchantSession,
    CoreIPCPKPaymentSetupFeature,
    CoreIPCPKContact,
    CoreIPCPKSecureElementPass,
    CoreIPCPKPayment,
    CoreIPCPKPaymentToken,
    CoreIPCPKShippingMethod,
    CoreIPCPKDateComponentsRange,
    CoreIPCCNContact,
    CoreIPCCNPhoneNumber,
    CoreIPCCNPostalAddress,
#endif
#if ENABLE(DATA_DETECTION) && HAVE(WK_SECURE_CODING_DATA_DETECTORS)
    CoreIPCDDScannerResult,
#if PLATFORM(MAC)
    CoreIPCDDSecureActionContext,
#endif
#endif
    CoreIPCDateComponents,
#if !HAVE(WK_SECURE_CODING_NSURLREQUEST)
    CoreIPCSecureCoding,
#endif // HAVE(WK_SECURE_CODING_NSURLREQUEST)
    CoreIPCString,
    CoreIPCURL
>;

class CoreIPCNSCFObject {
    WTF_MAKE_FAST_ALLOCATED;
public:
    CoreIPCNSCFObject(id);
    CoreIPCNSCFObject(UniqueRef<ObjectValue>&&);

    RetainPtr<id> toID() const;

    static bool valueIsAllowed(IPC::Decoder&, ObjectValue&);

    const UniqueRef<ObjectValue>& value() const { return m_value; }
private:
    UniqueRef<ObjectValue> m_value;
};

} // namespace WebKit

namespace IPC {

// This ArgumentCoders specialization for UniqueRef<ObjectValue> is to allow us to use
// makeUniqueRefWithoutFastMallocCheck<>, since we can't make the variant fast malloc'ed
template<> struct ArgumentCoder<UniqueRef<WebKit::ObjectValue>> {
    static void encode(Encoder&, const UniqueRef<WebKit::ObjectValue>&);
    static std::optional<UniqueRef<WebKit::ObjectValue>> decode(Decoder&);
};

} // namespace IPC

#endif // PLATFORM(COCOA)
