/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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

#if HAVE(CONTACTS)

#include "CoreIPCDateComponents.h"
#include "CoreIPCString.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS CNContact;
OBJC_CLASS CNPhoneNumber;
OBJC_CLASS CNPostalAddress;

namespace WebKit {

class CoreIPCCNPostalAddress {
public:
    CoreIPCCNPostalAddress(CNPostalAddress *);

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCCNPostalAddress, void>;

    CoreIPCCNPostalAddress(String&& street, String&& subLocality, String&& city, String&& subAdministrativeArea, String&& state, String&& postalCode, String&& country, String&& isoCountryCode, String&& formattedAddress)
        : m_street(WTFMove(street))
        , m_subLocality(WTFMove(subLocality))
        , m_city(WTFMove(city))
        , m_subAdministrativeArea(WTFMove(subAdministrativeArea))
        , m_state(WTFMove(state))
        , m_postalCode(WTFMove(postalCode))
        , m_country(WTFMove(country))
        , m_isoCountryCode(WTFMove(isoCountryCode))
        , m_formattedAddress(WTFMove(formattedAddress))
    {
    }

    String m_street;
    String m_subLocality;
    String m_city;
    String m_subAdministrativeArea;
    String m_state;
    String m_postalCode;
    String m_country;
    String m_isoCountryCode;
    String m_formattedAddress;
};

class CoreIPCCNPhoneNumber {
public:
    CoreIPCCNPhoneNumber(CNPhoneNumber *);

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCCNPhoneNumber, void>;

    CoreIPCCNPhoneNumber(String&& digits, String&& countryCode)
        : m_digits(WTFMove(digits))
        , m_countryCode(WTFMove(countryCode))
    {
    }

    String m_digits;
    String m_countryCode;
};

struct CoreIPCContactLabeledValue {
    String identifier;
    String label;
    std::variant<CoreIPCDateComponents, CoreIPCCNPhoneNumber, CoreIPCCNPostalAddress, CoreIPCString> value;

    template <typename T> static bool allValuesAreOfType(const Vector<CoreIPCContactLabeledValue>& values) {
        for (auto& value : values) {
            if (!std::holds_alternative<T>(value.value))
                return false;
        }
        return true;
    }
};

class CoreIPCCNContact {
public:
    CoreIPCCNContact(CNContact *);

    RetainPtr<id> toID() const;

    static bool isValidCNContactType(NSInteger);

private:
    friend struct IPC::ArgumentCoder<CoreIPCCNContact, void>;
    CoreIPCCNContact() = default;

    String m_identifier;
    NSInteger m_contactType { 0 };

    String m_namePrefix;
    String m_givenName;
    String m_middleName;
    String m_familyName;
    String m_previousFamilyName;
    String m_nameSuffix;
    String m_nickname;
    String m_organizationName;
    String m_departmentName;
    String m_jobTitle;
    String m_phoneticGivenName;
    String m_phoneticMiddleName;
    String m_phoneticFamilyName;
    String m_phoneticOrganizationName;
    String m_note;

    std::optional<CoreIPCDateComponents> m_birthday;
    std::optional<CoreIPCDateComponents> m_nonGregorianBirthday;

    Vector<CoreIPCContactLabeledValue> m_dates;
    Vector<CoreIPCContactLabeledValue> m_phoneNumbers;
    Vector<CoreIPCContactLabeledValue> m_emailAddresses;
    Vector<CoreIPCContactLabeledValue> m_postalAddresses;
    Vector<CoreIPCContactLabeledValue> m_urlAddresses;
};

} // namespace WebKit

#endif // HAVE(CONTACTS)

