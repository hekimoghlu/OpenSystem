/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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

#if ENABLE(WEB_AUTHN)

#include "BufferSource.h"
#include <wtf/KeyValuePair.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct AuthenticationExtensionsClientInputs {
    struct LargeBlobInputs {
        String support;
        std::optional<bool> read;
        std::optional<BufferSource> write;
    };

    struct PRFValues {
        BufferSource first;
        std::optional<BufferSource> second;
    };

    struct PRFInputs {
        std::optional<AuthenticationExtensionsClientInputs::PRFValues> eval;
        std::optional<Vector<KeyValuePair<String, AuthenticationExtensionsClientInputs::PRFValues>>> evalByCredential;
    };

    String appid;
    std::optional<bool> credProps;
    std::optional<AuthenticationExtensionsClientInputs::LargeBlobInputs> largeBlob;
    std::optional<AuthenticationExtensionsClientInputs::PRFInputs> prf;

    WEBCORE_EXPORT Vector<uint8_t> toCBOR() const;
    WEBCORE_EXPORT static std::optional<AuthenticationExtensionsClientInputs> fromCBOR(std::span<const uint8_t>);
};

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
