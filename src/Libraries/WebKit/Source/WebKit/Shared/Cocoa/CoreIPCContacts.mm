/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#import "CoreIPCContacts.h"

#if HAVE(CONTACTS)

#import <pal/spi/cocoa/ContactsSPI.h>
#import <pal/cocoa/ContactsSoftLink.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {

CoreIPCCNPhoneNumber::CoreIPCCNPhoneNumber(CNPhoneNumber *cnPhoneNumber)
    : m_digits(cnPhoneNumber.digits)
    , m_countryCode(cnPhoneNumber.countryCode)
{
}

RetainPtr<id> CoreIPCCNPhoneNumber::toID() const
{
    return [PAL::getCNPhoneNumberClass() phoneNumberWithDigits:(NSString *)m_digits countryCode:(NSString *)m_countryCode];
}

CoreIPCCNPostalAddress::CoreIPCCNPostalAddress(CNPostalAddress *cnPostalAddress)
    : m_street(cnPostalAddress.street)
    , m_subLocality(cnPostalAddress.subLocality)
    , m_city(cnPostalAddress.city)
    , m_subAdministrativeArea(cnPostalAddress.subAdministrativeArea)
    , m_state(cnPostalAddress.state)
    , m_postalCode(cnPostalAddress.postalCode)
    , m_country(cnPostalAddress.country)
    , m_isoCountryCode(cnPostalAddress.ISOCountryCode)
    , m_formattedAddress(cnPostalAddress.formattedAddress)
{
}

RetainPtr<id> CoreIPCCNPostalAddress::toID() const
{
    RetainPtr<CNMutablePostalAddress> address = adoptNS([[PAL::getCNMutablePostalAddressClass() alloc] init]);

    address.get().street = nsStringNilIfNull(m_street);
    address.get().subLocality = nsStringNilIfNull(m_subLocality);
    address.get().city = nsStringNilIfNull(m_city);
    address.get().subAdministrativeArea = nsStringNilIfNull(m_subAdministrativeArea);
    address.get().state = nsStringNilIfNull(m_state);
    address.get().postalCode = nsStringNilIfNull(m_postalCode);
    address.get().country = nsStringNilIfNull(m_country);
    address.get().ISOCountryCode = nsStringNilIfNull(m_isoCountryCode);
    address.get().formattedAddress = nsStringNilIfNull(m_formattedAddress);

    return address;
}

CoreIPCCNContact::CoreIPCCNContact(CNContact *contact)
{
    m_identifier = contact.identifier;
    m_contactType = contact.contactType;

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactNamePrefixKey()] && contact.namePrefix)
        m_namePrefix = contact.namePrefix;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactGivenNameKey()] && contact.givenName)
        m_givenName = contact.givenName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactMiddleNameKey()] && contact.middleName)
        m_middleName = contact.middleName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactFamilyNameKey()] && contact.familyName)
        m_familyName = contact.familyName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactPreviousFamilyNameKey()] && contact.previousFamilyName)
        m_previousFamilyName = contact.previousFamilyName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactNameSuffixKey()] && contact.nameSuffix)
        m_nameSuffix = contact.nameSuffix;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactNicknameKey()] && contact.nickname)
        m_nickname = contact.nickname;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactOrganizationNameKey()] && contact.organizationName)
        m_organizationName = contact.organizationName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactDepartmentNameKey()] && contact.departmentName)
        m_departmentName = contact.departmentName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactJobTitleKey()] && contact.jobTitle)
        m_jobTitle = contact.jobTitle;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactPhoneticGivenNameKey()] && contact.phoneticGivenName)
        m_phoneticGivenName = contact.phoneticGivenName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactPhoneticMiddleNameKey()] && contact.phoneticMiddleName)
        m_phoneticMiddleName = contact.phoneticMiddleName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactPhoneticFamilyNameKey()] && contact.phoneticFamilyName)
        m_phoneticFamilyName = contact.phoneticFamilyName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactPhoneticOrganizationNameKey()] && contact.phoneticOrganizationName)
        m_phoneticOrganizationName = contact.phoneticOrganizationName;
    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactNoteKey()] && contact.note)
        m_note = contact.note;

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactBirthdayKey()] && contact.birthday)
        m_birthday = contact.birthday;

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactNonGregorianBirthdayKey()] && contact.nonGregorianBirthday)
        m_nonGregorianBirthday = contact.nonGregorianBirthday;

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactDatesKey()] && contact.dates) {
        for (CNLabeledValue *labeledValue in contact.dates)
            m_dates.append({ labeledValue.identifier, labeledValue.label, CoreIPCDateComponents(labeledValue.value) });
    }

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactPhoneNumbersKey()] && contact.phoneNumbers) {
        for (CNLabeledValue *labeledValue in contact.phoneNumbers)
            m_phoneNumbers.append({ labeledValue.identifier, labeledValue.label, CoreIPCCNPhoneNumber(labeledValue.value) });
    }

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactEmailAddressesKey()] && contact.emailAddresses) {
        for (CNLabeledValue *labeledValue in contact.emailAddresses)
            m_emailAddresses.append({ labeledValue.identifier, labeledValue.label, (NSString *)labeledValue.value });
    }

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactPostalAddressesKey()] && contact.postalAddresses) {
        for (CNLabeledValue *labeledValue in contact.postalAddresses)
            m_postalAddresses.append({ labeledValue.identifier, labeledValue.label, CoreIPCCNPostalAddress(labeledValue.value) });
    }

    if ([contact isKeyAvailable:PAL::get_Contacts_CNContactUrlAddressesKey()] && contact.urlAddresses) {
        for (CNLabeledValue *labeledValue in contact.urlAddresses)
            m_urlAddresses.append({ labeledValue.identifier, labeledValue.label, (NSString *)labeledValue.value });
    }
}

bool CoreIPCCNContact::isValidCNContactType(NSInteger proposedType)
{
    return proposedType == CNContactTypePerson || proposedType == CNContactTypeOrganization;
}

static RetainPtr<NSArray> nsArrayFromVectorOfLabeledValues(const Vector<CoreIPCContactLabeledValue>& labeledValues)
{
    return createNSArray(labeledValues, [] (auto& labeledValue) -> RetainPtr<id> {
        auto theValue = std::visit([] (auto& actualValue) -> RetainPtr<id> {
            return actualValue.toID();
        }, labeledValue.value);

        return adoptNS([[PAL::getCNLabeledValueClass() alloc] initWithIdentifier:labeledValue.identifier label:labeledValue.label value:theValue.get()]);
    });
}

RetainPtr<id> CoreIPCCNContact::toID() const
{
    RetainPtr<CNMutableContact> result = adoptNS([[PAL::getCNMutableContactClass() alloc] initWithIdentifier:m_identifier]);
    result.get().contactType = (CNContactType)m_contactType;

    if (!m_namePrefix.isNull())
        result.get().namePrefix = m_namePrefix;
    if (!m_givenName.isNull())
        result.get().givenName = m_givenName;
    if (!m_middleName.isNull())
        result.get().middleName = m_middleName;
    if (!m_familyName.isNull())
        result.get().familyName = m_familyName;
    if (!m_previousFamilyName.isNull())
        result.get().previousFamilyName = m_previousFamilyName;
    if (!m_nameSuffix.isNull())
        result.get().nameSuffix = m_nameSuffix;
    if (!m_nickname.isNull())
        result.get().nickname = m_nickname;
    if (!m_organizationName.isNull())
        result.get().organizationName = m_organizationName;
    if (!m_departmentName.isNull())
        result.get().departmentName = m_departmentName;
    if (!m_jobTitle.isNull())
        result.get().jobTitle = m_jobTitle;
    if (!m_phoneticGivenName.isNull())
        result.get().phoneticGivenName = m_phoneticGivenName;
    if (!m_phoneticMiddleName.isNull())
        result.get().phoneticMiddleName = m_phoneticMiddleName;
    if (!m_phoneticFamilyName.isNull())
        result.get().phoneticFamilyName = m_phoneticFamilyName;
    if (!m_phoneticOrganizationName.isNull())
        result.get().phoneticOrganizationName = m_phoneticOrganizationName;
    if (!m_note.isNull())
        result.get().note = m_note;

    if (m_birthday)
        result.get().birthday = m_birthday->toID().get();
    if (m_nonGregorianBirthday)
        result.get().nonGregorianBirthday = m_nonGregorianBirthday->toID().get();

    if (!m_dates.isEmpty())
        result.get().dates = nsArrayFromVectorOfLabeledValues(m_dates).get();
    if (!m_phoneNumbers.isEmpty())
        result.get().phoneNumbers = nsArrayFromVectorOfLabeledValues(m_phoneNumbers).get();
    if (!m_emailAddresses.isEmpty()) {
        auto emailAddresses = nsArrayFromVectorOfLabeledValues(m_emailAddresses);

        result.get().emailAddresses = emailAddresses.get();
    }
    if (!m_postalAddresses.isEmpty()) {
        auto postal = nsArrayFromVectorOfLabeledValues(m_postalAddresses);
        result.get().postalAddresses = postal.get();

    }
    if (!m_urlAddresses.isEmpty())
        result.get().urlAddresses = nsArrayFromVectorOfLabeledValues(m_urlAddresses).get();

    return result;
}

} // namespace WebKit

#endif // HAVE(CONTACTS)

