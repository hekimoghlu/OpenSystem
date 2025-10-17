/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#ifndef Credential_h
#define Credential_h

#include <wtf/Platform.h>

#if PLATFORM(COCOA)
#include "CredentialCocoa.h"
#elif USE(SOUP)
#include "CredentialSoup.h"
#else

#include "CredentialBase.h"

namespace WebCore {

class Credential : public CredentialBase {
public:
    Credential()
        : CredentialBase()
    {
    }

    Credential(const String& user, const String& password, CredentialPersistence persistence)
        : CredentialBase(user, password, persistence)
    {
    }

    Credential(NonPlatformData&& data)
        : CredentialBase(data.user, data.password, data.persistence)
    {
    }

    Credential(const Credential& original, CredentialPersistence persistence)
        : CredentialBase(original, persistence)
    {
    }
};

}

#endif

#endif // Credential_h
