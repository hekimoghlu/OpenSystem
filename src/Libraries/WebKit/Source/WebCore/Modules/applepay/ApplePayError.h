/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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

#if ENABLE(APPLE_PAY)

#include "ApplePayErrorCode.h"
#include "ApplePayErrorContactField.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ApplePayError final : public RefCounted<ApplePayError> {
public:

    enum class Domain : uint8_t {
        Disbursement
    };

    static Ref<ApplePayError> create(ApplePayErrorCode code, std::optional<ApplePayErrorContactField> contactField, const String& message, std::optional<ApplePayError::Domain> domain = { })
    {
        return adoptRef(*new ApplePayError(code, contactField, message, domain));
    }

    virtual ~ApplePayError() = default;

    ApplePayErrorCode code() const { return m_code; }
    void setCode(ApplePayErrorCode code) { m_code = code; }

    std::optional<ApplePayErrorContactField> contactField() const { return m_contactField; }
    void setContactField(std::optional<ApplePayErrorContactField> contactField) { m_contactField = contactField; }

    String message() const { return m_message; }
    void setMessage(String&& message) { m_message = WTFMove(message); }

    std::optional<Domain> domain() const { return m_domain; }
    void setDomain(std::optional<Domain> domain) { m_domain = domain; }


private:
    ApplePayError(ApplePayErrorCode code, std::optional<ApplePayErrorContactField> contactField, const String& message, std::optional<ApplePayError::Domain> domain)
        : m_code(code)
        , m_contactField(contactField)
        , m_message(message)
        , m_domain(domain)
    {
    }

    ApplePayErrorCode m_code;
    std::optional<ApplePayErrorContactField> m_contactField;
    String m_message;

    std::optional<ApplePayError::Domain> m_domain;
};

} // namespace WebCore

#endif
