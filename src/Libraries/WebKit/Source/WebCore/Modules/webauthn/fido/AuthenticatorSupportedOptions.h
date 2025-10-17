/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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

// Copyright 2018 The Chromium Authors. All rights reserved.
// Copyright (C) 2018 Apple Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#if ENABLE(WEB_AUTHN)

#include "CBORValue.h"
#include <wtf/Forward.h>

namespace fido {

// Represents CTAP device properties and capabilities received as a response to
// AuthenticatorGetInfo command.
class WEBCORE_EXPORT AuthenticatorSupportedOptions {
    WTF_MAKE_NONCOPYABLE(AuthenticatorSupportedOptions);
public:
    enum class UserVerificationAvailability {
        // e.g. Authenticator with finger print sensor and user's fingerprint is
        // registered to the device.
        kSupportedAndConfigured,
        // e.g. Authenticator with fingerprint sensor without user's fingerprint
        // registered.
        kSupportedButNotConfigured,
        kNotSupported
    };

    enum class ResidentKeyAvailability : bool {
        kSupported,
        kNotSupported
    };

    enum class ClientPinAvailability {
        kSupportedAndPinSet,
        kSupportedButPinNotSet,
        kNotSupported,
    };

    AuthenticatorSupportedOptions() = default;
    AuthenticatorSupportedOptions(AuthenticatorSupportedOptions&&) = default;
    AuthenticatorSupportedOptions& operator=(AuthenticatorSupportedOptions&&) = default;

    AuthenticatorSupportedOptions& setIsPlatformDevice(bool);
    AuthenticatorSupportedOptions& setResidentKeyAvailability(ResidentKeyAvailability);
    AuthenticatorSupportedOptions& setUserVerificationAvailability(UserVerificationAvailability);
    AuthenticatorSupportedOptions& setUserPresenceRequired(bool);
    AuthenticatorSupportedOptions& setClientPinAvailability(ClientPinAvailability);

    bool isPlatformDevice() const { return m_isPlatformDevice; }
    ResidentKeyAvailability residentKeyAvailability() const { return m_residentKeyAvailability; };
    UserVerificationAvailability userVerificationAvailability() const { return m_userVerificationAvailability; };
    bool userPresenceRequired() const { return m_userPresenceRequired; }
    ClientPinAvailability clientPinAvailability() const { return m_clientPinAvailability; }

private:
    // Indicates that the device is attached to the client and therefore can't be
    // removed and used on another client.
    bool m_isPlatformDevice { false };
    // Indicates that the device is capable of storing keys on the device itself
    // and therefore can satisfy the authenticatorGetAssertion request with
    // allowList parameter not specified or empty.
    ResidentKeyAvailability m_residentKeyAvailability { ResidentKeyAvailability::kNotSupported };
    // Indicates whether the device is capable of verifying the user on its own.
    UserVerificationAvailability m_userVerificationAvailability { UserVerificationAvailability::kNotSupported };
    bool m_userPresenceRequired { true };
    // Represents whether client pin in set and stored in device. Set as null
    // optional if client pin capability is not supported by the authenticator.
    ClientPinAvailability m_clientPinAvailability { ClientPinAvailability::kNotSupported };
};

WEBCORE_EXPORT cbor::CBORValue convertToCBOR(const AuthenticatorSupportedOptions&);

} // namespace fido

#endif // ENABLE(WEB_AUTHN)
