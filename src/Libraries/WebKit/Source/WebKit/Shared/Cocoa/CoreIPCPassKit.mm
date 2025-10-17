/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
#import "config.h"
#import "CoreIPCPassKit.h"

#if USE(PASSKIT)

#import <pal/cocoa/PassKitSoftLink.h>

namespace WebKit {

CoreIPCPKContact::CoreIPCPKContact(PKContact *contact)
    : m_emailAddress(contact.emailAddress)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    , m_supplementarySublocality(contact.supplementarySubLocality)
ALLOW_DEPRECATED_DECLARATIONS_END
{
    if (contact.name)
        m_name = contact.name;
    if (contact.phoneNumber)
        m_phoneNumber = contact.phoneNumber;
    if (contact.postalAddress)
        m_postalAddress = contact.postalAddress;
}

RetainPtr<id> CoreIPCPKContact::toID() const
{
    RetainPtr<PKContact> contact = adoptNS([[PAL::getPKContactClass() alloc] init]);

    if (m_name)
        contact.get().name = (NSPersonNameComponents *)m_name->toID();
    if (m_phoneNumber)
        contact.get().phoneNumber = (CNPhoneNumber *)m_phoneNumber->toID();
    if (m_postalAddress)
        contact.get().postalAddress = (CNPostalAddress *)m_postalAddress->toID();

    contact.get().emailAddress = nsStringNilIfNull(m_emailAddress);
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    contact.get().supplementarySubLocality = nsStringNilIfNull(m_supplementarySublocality);
ALLOW_DEPRECATED_DECLARATIONS_END

    return contact;
}

} // namespace WebKit

#endif // USE(PASSKIT)
