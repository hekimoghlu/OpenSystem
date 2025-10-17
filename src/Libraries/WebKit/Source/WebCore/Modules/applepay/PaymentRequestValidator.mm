/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
#import "PaymentRequestValidator.h"

#if ENABLE(APPLE_PAY)

#import "ApplePaySessionPaymentRequest.h"
#import "ApplePayShippingMethod.h"
#import <unicode/ucurr.h>
#import <unicode/uloc.h>
#import <wtf/text/MakeString.h>
#import <wtf/unicode/icu/ICUHelpers.h>

namespace WebCore {

static ExceptionOr<void> validateCountryCode(const String&);
static ExceptionOr<void> validateCurrencyCode(const String&);
static ExceptionOr<void> validateMerchantCapabilities(const ApplePaySessionPaymentRequest::MerchantCapabilities&);
static ExceptionOr<void> validateSupportedNetworks(const Vector<String>&);
static ExceptionOr<void> validateShippingMethods(const Vector<ApplePayShippingMethod>&);
static ExceptionOr<void> validateShippingMethod(const ApplePayShippingMethod&);

ExceptionOr<void> PaymentRequestValidator::validate(const ApplePaySessionPaymentRequest& paymentRequest, OptionSet<Field> fieldsToValidate)
{
    if (fieldsToValidate.contains(Field::CountryCode)) {
        auto validatedCountryCode = validateCountryCode(paymentRequest.countryCode());
        if (validatedCountryCode.hasException())
            return validatedCountryCode.releaseException();
    }

    if (fieldsToValidate.contains(Field::CurrencyCode)) {
        auto validatedCurrencyCode = validateCurrencyCode(paymentRequest.currencyCode());
        if (validatedCurrencyCode.hasException())
            return validatedCurrencyCode.releaseException();
    }

    if (fieldsToValidate.contains(Field::SupportedNetworks)) {
        auto validatedSupportedNetworks = validateSupportedNetworks(paymentRequest.supportedNetworks());
        if (validatedSupportedNetworks.hasException())
            return validatedSupportedNetworks.releaseException();
    }

    if (fieldsToValidate.contains(Field::MerchantCapabilities)) {
        auto validatedMerchantCapabilities = validateMerchantCapabilities(paymentRequest.merchantCapabilities());
        if (validatedMerchantCapabilities.hasException())
            return validatedMerchantCapabilities.releaseException();
    }

    if (fieldsToValidate.contains(Field::Total)) {
        auto validatedTotal = validateTotal(paymentRequest.total());
        if (validatedTotal.hasException())
            return validatedTotal.releaseException();
    }

    if (fieldsToValidate.contains(Field::ShippingMethods)) {
        auto validatedShippingMethods = validateShippingMethods(paymentRequest.shippingMethods());
        if (validatedShippingMethods.hasException())
            return validatedShippingMethods.releaseException();
    }

    if (fieldsToValidate.contains(Field::CountryCode)) {
        for (auto& countryCode : paymentRequest.supportedCountries()) {
            auto validatedCountryCode = validateCountryCode(countryCode);
            if (validatedCountryCode.hasException())
                return validatedCountryCode.releaseException();
        }
    }

    return { };
}

ExceptionOr<void> PaymentRequestValidator::validateTotal(const ApplePayLineItem& total)
{
    if (!total.label)
        return Exception { ExceptionCode::TypeError, "Missing total label."_s };

    if (!total.amount)
        return Exception { ExceptionCode::TypeError, "Missing total amount."_s };

    double amount = [NSDecimalNumber decimalNumberWithString:total.amount locale:@{ NSLocaleDecimalSeparator : @"." }].doubleValue;

    if (amount < 0)
        return Exception { ExceptionCode::TypeError, "Total amount must not be negative."_s };

    // We can safely defer a maximum amount check to the underlying payment system, instead.
    // The downside is we lose an informative error mode and get an opaque payment sheet error for too large total amounts.
    // FIXME: <https://webkit.org/b/276088> PaymentRequestValidator should adopt per-currency checks for total amounts.

    return { };
}

static ExceptionOr<void> validateCountryCode(const String& countryCode)
{
    if (!countryCode)
        return Exception { ExceptionCode::TypeError, "Missing country code."_s };

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    for (auto* countryCodePtr = uloc_getISOCountries(); *countryCodePtr; ++countryCodePtr) {
        if (countryCode == StringView::fromLatin1(*countryCodePtr))
            return { };
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    return Exception { ExceptionCode::TypeError, makeString("\""_s, countryCode, "\" is not a valid country code."_s) };
}

static ExceptionOr<void> validateCurrencyCode(const String& currencyCode)
{
    if (!currencyCode)
        return Exception { ExceptionCode::TypeError, "Missing currency code."_s };

    UErrorCode errorCode = U_ZERO_ERROR;
    auto currencyCodes = std::unique_ptr<UEnumeration, ICUDeleter<uenum_close>>(ucurr_openISOCurrencies(UCURR_ALL, &errorCode));

    int32_t length;
    while (auto *currencyCodePtr = uenum_next(currencyCodes.get(), &length, &errorCode)) {
        if (currencyCode == StringView::fromLatin1(currencyCodePtr))
            return { };
    }

    return Exception { ExceptionCode::TypeError, makeString("\""_s, currencyCode, "\" is not a valid currency code."_s) };
}

static ExceptionOr<void> validateMerchantCapabilities(const ApplePaySessionPaymentRequest::MerchantCapabilities& merchantCapabilities)
{
    if (!merchantCapabilities.supports3DS && !merchantCapabilities.supportsEMV && !merchantCapabilities.supportsCredit && !merchantCapabilities.supportsDebit)
        return Exception { ExceptionCode::TypeError, "Missing merchant capabilities."_s };

    return { };
}

static ExceptionOr<void> validateSupportedNetworks(const Vector<String>& supportedNetworks)
{
    if (supportedNetworks.isEmpty())
        return Exception { ExceptionCode::TypeError, "Missing supported networks."_s };

    return { };
}

static ExceptionOr<void> validateShippingMethod(const ApplePayShippingMethod& shippingMethod)
{
    NSDecimalNumber *amount = [NSDecimalNumber decimalNumberWithString:shippingMethod.amount locale:@{ NSLocaleDecimalSeparator : @"." }];
    if (amount.integerValue < 0)
        return Exception { ExceptionCode::TypeError, "Shipping method amount must be greater than or equal to zero."_s };

    return { };
}

static ExceptionOr<void> validateShippingMethods(const Vector<ApplePayShippingMethod>& shippingMethods)
{
    for (const auto& shippingMethod : shippingMethods) {
        auto validatedShippingMethod = validateShippingMethod(shippingMethod);
        if (validatedShippingMethod.hasException())
            return validatedShippingMethod.releaseException();
    }

    return { };
}

}

#endif
