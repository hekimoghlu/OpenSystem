/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#import "PaymentMethod.h"

#if ENABLE(APPLE_PAY)

#import "ApplePayPaymentMethod.h"
#import "ApplePayPaymentMethodType.h"
#import <pal/spi/cocoa/PassKitSPI.h>

namespace WebCore {

static void finishConverting(PKPaymentMethod *paymentMethod, ApplePayPaymentMethod& result)
{
#if HAVE(PASSKIT_INSTALLMENTS)
    if (NSString *bindToken = paymentMethod.bindToken)
        result.bindToken = bindToken;
#else
    UNUSED_PARAM(paymentMethod);
    UNUSED_PARAM(result);
#endif
}

static ApplePayPaymentPass::ActivationState convert(PKPaymentPassActivationState paymentPassActivationState)
{
    switch (paymentPassActivationState) {
    case PKPaymentPassActivationStateActivated:
        return ApplePayPaymentPass::ActivationState::Activated;
    case PKPaymentPassActivationStateRequiresActivation:
        return ApplePayPaymentPass::ActivationState::RequiresActivation;
    case PKPaymentPassActivationStateActivating:
        return ApplePayPaymentPass::ActivationState::Activating;
    case PKPaymentPassActivationStateSuspended:
        return ApplePayPaymentPass::ActivationState::Suspended;
    case PKPaymentPassActivationStateDeactivated:
        return ApplePayPaymentPass::ActivationState::Deactivated;
    }
}

static std::optional<ApplePayPaymentPass> convert(PKPaymentPass *paymentPass)
{
    if (!paymentPass)
        return std::nullopt;

    ApplePayPaymentPass result;

    result.primaryAccountIdentifier = paymentPass.primaryAccountIdentifier;
    result.primaryAccountNumberSuffix = paymentPass.primaryAccountNumberSuffix;

    if (NSString *deviceAccountIdentifier = paymentPass.deviceAccountIdentifier)
        result.deviceAccountIdentifier = deviceAccountIdentifier;
    if (NSString *deviceAccountNumberSuffix = paymentPass.deviceAccountNumberSuffix)
        result.deviceAccountNumberSuffix = deviceAccountNumberSuffix;

    result.activationState = convert(paymentPass.activationState);

    return result;
}

static std::optional<ApplePayPaymentMethod::Type> convert(PKPaymentMethodType paymentMethodType)
{
    switch (paymentMethodType) {
    case PKPaymentMethodTypeDebit:
        return ApplePayPaymentMethod::Type::Debit;
    case PKPaymentMethodTypeCredit:
        return ApplePayPaymentMethod::Type::Credit;
    case PKPaymentMethodTypePrepaid:
        return ApplePayPaymentMethod::Type::Prepaid;
    case PKPaymentMethodTypeStore:
        return ApplePayPaymentMethod::Type::Store;
    case PKPaymentMethodTypeUnknown:
    default:
        return std::nullopt;
    }
}

static void convert(CNLabeledValue<CNPostalAddress*> *postalAddress, ApplePayPaymentContact &result)
{
    if (NSString *street = postalAddress.value.street)
        result.addressLines = { String { street } };
    result.subLocality = postalAddress.value.subLocality;
    result.locality = postalAddress.value.city;
    result.subAdministrativeArea = postalAddress.value.subAdministrativeArea;
    result.administrativeArea = postalAddress.value.state;
    result.postalCode = postalAddress.value.postalCode;
    result.country = postalAddress.value.country;
    result.countryCode = postalAddress.value.ISOCountryCode;
}

static std::optional<ApplePayPaymentContact> convert(CNContact *billingContact)
{
    if (!billingContact)
        return std::nullopt;

    ApplePayPaymentContact result;
    
    if (auto firstPhoneNumber = billingContact.phoneNumbers.firstObject)
        result.phoneNumber = firstPhoneNumber.value.stringValue;
    
    if (auto firstEmailAddress = billingContact.emailAddresses.firstObject)
        result.emailAddress = firstEmailAddress.value;
    
    result.givenName = billingContact.givenName;
    result.familyName = billingContact.familyName;
    
    result.phoneticGivenName = billingContact.phoneticGivenName;
    result.phoneticFamilyName = billingContact.phoneticFamilyName;
    
    if (CNLabeledValue<CNPostalAddress*> *firstPostalAddress = billingContact.postalAddresses.firstObject)
        convert(firstPostalAddress, result);

    return result;
}

static ApplePayPaymentMethod convert(PKPaymentMethod *paymentMethod)
{
    ApplePayPaymentMethod result;
    
    if (NSString *displayName = paymentMethod.displayName)
        result.displayName = displayName;
    if (NSString *network = paymentMethod.network)
        result.network = network;
    result.billingContact = convert(paymentMethod.billingAddress);
    result.type = convert(paymentMethod.type);
    result.paymentPass = convert(paymentMethod.paymentPass);

    finishConverting(paymentMethod, result);

    return result;
}

PaymentMethod::PaymentMethod() = default;

PaymentMethod::PaymentMethod(RetainPtr<PKPaymentMethod>&& pkPaymentMethod)
    : m_pkPaymentMethod { WTFMove(pkPaymentMethod) }
{
}

PaymentMethod::~PaymentMethod() = default;

ApplePayPaymentMethod PaymentMethod::toApplePayPaymentMethod() const
{
    return convert(m_pkPaymentMethod.get());
}

PKPaymentMethod *PaymentMethod::pkPaymentMethod() const
{
    return m_pkPaymentMethod.get();
}

}

#endif
