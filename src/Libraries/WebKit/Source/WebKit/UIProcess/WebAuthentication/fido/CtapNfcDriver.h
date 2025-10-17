/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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

#if ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)

#include "CtapDriver.h"
#include "NfcConnection.h"
#include <wtf/UniqueRef.h>

namespace WebKit {

// The following implements the CTAP NFC protocol:
// https://fidoalliance.org/specs/fido-v2.0-ps-20190130/fido-client-to-authenticator-protocol-v2.0-ps-20190130.html#nfc
class CtapNfcDriver final : public CtapDriver {
public:
    static Ref<CtapNfcDriver> create(Ref<NfcConnection>&&);

    void transact(Vector<uint8_t>&& data, ResponseCallback&&) final;

private:
    explicit CtapNfcDriver(Ref<NfcConnection>&&);

    void respondAsync(ResponseCallback&&, Vector<uint8_t>&& response) const;

    Ref<NfcConnection> m_connection;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN) && HAVE(NEAR_FIELD)
