/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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

#if USE(PASSKIT)

#include "CoreIPCContacts.h"
#include "CoreIPCPersonNameComponents.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS PKContact;

namespace WebKit {

class CoreIPCPKContact {
public:
    CoreIPCPKContact(PKContact *);

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCPKContact, void>;

    CoreIPCPKContact(std::optional<CoreIPCPersonNameComponents>&& name, String&& emailAddress, std::optional<CoreIPCCNPhoneNumber>&& phoneNumber, std::optional<CoreIPCCNPostalAddress>&& postalAddress, String&& supplementarySublocality)
        : m_name(WTFMove(name))
        , m_emailAddress(WTFMove(emailAddress))
        , m_phoneNumber(WTFMove(phoneNumber))
        , m_postalAddress(WTFMove(postalAddress))
        , m_supplementarySublocality(WTFMove(supplementarySublocality))
    {
    }

    std::optional<CoreIPCPersonNameComponents> m_name;
    String m_emailAddress;
    std::optional<CoreIPCCNPhoneNumber> m_phoneNumber;
    std::optional<CoreIPCCNPostalAddress> m_postalAddress;
    String m_supplementarySublocality;
};

} // namespace WebKit

#endif // USE(PASSKIT)
