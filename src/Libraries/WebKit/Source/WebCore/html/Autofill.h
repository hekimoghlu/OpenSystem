/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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

#include <wtf/text/AtomString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class AutofillMantle {
    Expectation,
    Anchor
};

enum class NonAutofillCredentialType : bool {
    None,
    WebAuthn
};

enum class AutofillFieldName : uint8_t {
    None,
    Name,
    HonorificPrefix,
    GivenName,
    AdditionalName,
    FamilyName,
    HonorificSuffix,
    Nickname,
    Username,
    NewPassword,
    CurrentPassword,
    OrganizationTitle,
    Organization,
    StreetAddress,
    AddressLine1,
    AddressLine2,
    AddressLine3,
    AddressLevel4,
    AddressLevel3,
    AddressLevel2,
    AddressLevel1,
    Country,
    CountryName,
    PostalCode,
    CcName,
    CcGivenName,
    CcAdditionalName,
    CcFamilyName,
    CcNumber,
    CcExp,
    CcExpMonth,
    CcExpYear,
    CcCsc,
    CcType,
    TransactionCurrency,
    TransactionAmount,
    Language,
    Bday,
    BdayDay,
    BdayMonth,
    BdayYear,
    Sex,
    URL,
    Photo,
    Tel,
    TelCountryCode,
    TelNational,
    TelAreaCode,
    TelLocal,
    TelLocalPrefix,
    TelLocalSuffix,
    TelExtension,
    Email,
    Impp,
    WebAuthn,
    OneTimeCode,
    DeviceEID,
    DeviceIMEI,
};

WEBCORE_EXPORT AutofillFieldName toAutofillFieldName(const AtomString&);
WEBCORE_EXPORT String nonAutofillCredentialTypeString(NonAutofillCredentialType);

class HTMLFormControlElement;

class AutofillData {
public:
    static AutofillData createFromHTMLFormControlElement(const HTMLFormControlElement&);

    AutofillData(const AtomString& fieldName, const String& idlExposedValue, NonAutofillCredentialType nonAutofillCredentialType)
        : fieldName(fieldName)
        , idlExposedValue(idlExposedValue)
        , nonAutofillCredentialType(nonAutofillCredentialType)
    {
    }

    // We could add support for hint tokens and scope tokens if those ever became useful to anyone.

    AtomString fieldName;
    String idlExposedValue;
    NonAutofillCredentialType nonAutofillCredentialType;
};

} // namespace WebCore
