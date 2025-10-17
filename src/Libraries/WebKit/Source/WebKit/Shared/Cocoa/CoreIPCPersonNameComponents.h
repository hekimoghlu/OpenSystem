/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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

#if PLATFORM(COCOA)

#include "ArgumentCodersCocoa.h"
#include <Foundation/Foundation.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class CoreIPCPersonNameComponents {
WTF_MAKE_TZONE_ALLOCATED(CoreIPCPersonNameComponents);
public:
    CoreIPCPersonNameComponents(NSPersonNameComponents *);

    RetainPtr<id> toID() const;

private:
    friend struct IPC::ArgumentCoder<CoreIPCPersonNameComponents, void>;

    CoreIPCPersonNameComponents(const String& namePrefix, const String& givenName, const String& middleName, const String& familyName, const String& nickname, std::unique_ptr<CoreIPCPersonNameComponents>&& phoneticRepresentation)
        : m_namePrefix(namePrefix)
        , m_givenName(givenName)
        , m_middleName(middleName)
        , m_familyName(familyName)
        , m_nickname(nickname)
        , m_phoneticRepresentation(WTFMove(phoneticRepresentation))
    {
    }

    String m_namePrefix;
    String m_givenName;
    String m_middleName;
    String m_familyName;
    String m_nickname;
    std::unique_ptr<CoreIPCPersonNameComponents> m_phoneticRepresentation;
};

} // namespace WebKit

#endif // PLATFORM(COCOA)
