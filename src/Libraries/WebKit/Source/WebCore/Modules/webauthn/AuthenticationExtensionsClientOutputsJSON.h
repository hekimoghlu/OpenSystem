/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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
#include "CredentialPropertiesOutput.h"

namespace WebCore {

struct AuthenticationExtensionsClientOutputsJSON {
    struct LargeBlobOutputsJSON {
        std::optional<bool> supported;
        String blob;
        std::optional<bool> written;
    };
    struct PRFValuesJSON {
        String first;
        String second;
    };
    struct PRFOutputsJSON {
        std::optional<bool> enabled;
        std::optional<PRFValuesJSON> results;
    };
    std::optional<bool> appid;
    std::optional<CredentialPropertiesOutput> credProps;
    std::optional<LargeBlobOutputsJSON> largeBlob;
    std::optional<PRFOutputsJSON> prf;
};
}

#endif // ENABLE(WEB_AUTHN)
