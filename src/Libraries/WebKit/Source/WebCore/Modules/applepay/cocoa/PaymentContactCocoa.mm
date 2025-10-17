/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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
#import "PaymentContact.h"

#if ENABLE(APPLE_PAY)

#import "ApplePayPaymentContact.h"
#import <Contacts/Contacts.h>
#import <pal/cocoa/PassKitSoftLink.h>
#import <wtf/SoftLinking.h>
#import <wtf/text/StringBuilder.h>

SOFT_LINK_FRAMEWORK(Contacts)
SOFT_LINK_CLASS(Contacts, CNMutablePostalAddress)
SOFT_LINK_CLASS(Contacts, CNPhoneNumber)

namespace WebCore {

static RetainPtr<PKContact> convert(unsigned version, const ApplePayPaymentContact& contact)
{
    auto result = adoptNS([PAL::allocPKContactInstance() init]);

    NSString *familyName = nil;
    NSString *phoneticFamilyName = nil;
    if (!contact.familyName.isEmpty()) {
        familyName = contact.familyName;
        if (version >= 3 && !contact.phoneticFamilyName.isEmpty())
            phoneticFamilyName = contact.phoneticFamilyName;
    }

    NSString *givenName = nil;
    NSString *phoneticGivenName = nil;
    if (!contact.givenName.isEmpty()) {
        givenName = contact.givenName;
        if (version >= 3 && !contact.phoneticGivenName.isEmpty())
            phoneticGivenName = contact.phoneticGivenName;
    }

    if (familyName || givenName) {
        auto name = adoptNS([[NSPersonNameComponents alloc] init]);
        [name setFamilyName:familyName];
        [name setGivenName:givenName];
        if (phoneticFamilyName || phoneticGivenName) {
            auto phoneticName = adoptNS([[NSPersonNameComponents alloc] init]);
            [phoneticName setFamilyName:phoneticFamilyName];
            [phoneticName setGivenName:phoneticGivenName];
            [name setPhoneticRepresentation:phoneticName.get()];
        }
        [result setName:name.get()];
    }

    if (!contact.emailAddress.isEmpty())
        [result setEmailAddress:contact.emailAddress];

    if (!contact.phoneNumber.isEmpty())
        [result setPhoneNumber:adoptNS([allocCNPhoneNumberInstance() initWithStringValue:contact.phoneNumber]).get()];

    if (contact.addressLines && !contact.addressLines->isEmpty()) {
        auto address = adoptNS([allocCNMutablePostalAddressInstance() init]);

        StringBuilder builder;
        for (unsigned i = 0; i < contact.addressLines->size(); ++i) {
            builder.append(contact.addressLines->at(i));
            if (i != contact.addressLines->size() - 1)
                builder.append('\n');
        }

        // FIXME: StringBuilder should hava a toNSString() function to avoid the extra String allocation.
        [address setStreet:builder.toString()];

        if (!contact.subLocality.isEmpty())
            [address setSubLocality:contact.subLocality];
        if (!contact.locality.isEmpty())
            [address setCity:contact.locality];
        if (!contact.postalCode.isEmpty())
            [address setPostalCode:contact.postalCode];
        if (!contact.subAdministrativeArea.isEmpty())
            [address setSubAdministrativeArea:contact.subAdministrativeArea];
        if (!contact.administrativeArea.isEmpty())
            [address setState:contact.administrativeArea];
        if (!contact.country.isEmpty())
            [address setCountry:contact.country];
        if (!contact.countryCode.isEmpty())
            [address setISOCountryCode:contact.countryCode];

        [result setPostalAddress:address.get()];
    }

    return result;
}

static ApplePayPaymentContact convert(unsigned version, PKContact *contact)
{
    ASSERT(contact);

    ApplePayPaymentContact result;

    result.phoneNumber = static_cast<CNPhoneNumber *>(contact.phoneNumber).stringValue;
    result.emailAddress = contact.emailAddress;

    NSPersonNameComponents *name = contact.name;
    result.givenName = name.givenName;
    result.familyName = name.familyName;
    if (name)
        result.localizedName = [NSPersonNameComponentsFormatter localizedStringFromPersonNameComponents:name style:NSPersonNameComponentsFormatterStyleDefault options:0];

    if (version >= 3) {
        NSPersonNameComponents *phoneticName = name.phoneticRepresentation;
        result.phoneticGivenName = phoneticName.givenName;
        result.phoneticFamilyName = phoneticName.familyName;
        if (phoneticName)
            result.localizedPhoneticName = [NSPersonNameComponentsFormatter localizedStringFromPersonNameComponents:name style:NSPersonNameComponentsFormatterStyleDefault options:NSPersonNameComponentsFormatterPhonetic];
    }

    CNPostalAddress *postalAddress = contact.postalAddress;
    if (postalAddress.street.length)
        result.addressLines = String(postalAddress.street).split('\n');
    result.subLocality = postalAddress.subLocality;
    result.locality = postalAddress.city;
    result.postalCode = postalAddress.postalCode;
    result.subAdministrativeArea = postalAddress.subAdministrativeArea;
    result.administrativeArea = postalAddress.state;
    result.country = postalAddress.country;
    result.countryCode = postalAddress.ISOCountryCode;

    return result;
}

PaymentContact::PaymentContact() = default;

PaymentContact::PaymentContact(RetainPtr<PKContact>&& pkContact)
    : m_pkContact { WTFMove(pkContact) }
{
}

PaymentContact::~PaymentContact() = default;

PaymentContact PaymentContact::fromApplePayPaymentContact(unsigned version, const ApplePayPaymentContact& contact)
{
    return PaymentContact(convert(version, contact).get());
}

ApplePayPaymentContact PaymentContact::toApplePayPaymentContact(unsigned version) const
{
    return convert(version, m_pkContact.get());
}

RetainPtr<PKContact> PaymentContact::pkContact() const
{
    return m_pkContact;
}

}

#endif
