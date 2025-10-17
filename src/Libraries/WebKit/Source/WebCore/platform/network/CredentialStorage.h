/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#include "Credential.h"
#include "ProtectionSpaceHash.h"
#include "SecurityOriginData.h"
#include <wtf/HashMap.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ProtectionSpace;

class CredentialStorage {
public:
    // WebCore session credential storage.
    WEBCORE_EXPORT void set(const String&, const Credential&, const ProtectionSpace&, const URL&);
    WEBCORE_EXPORT Credential get(const String&, const ProtectionSpace&);
    WEBCORE_EXPORT void remove(const String&, const ProtectionSpace&);
    WEBCORE_EXPORT void removeCredentialsWithOrigin(const SecurityOriginData&);
    WEBCORE_EXPORT void clearCredentials();

#if PLATFORM(COCOA)
    // OS credential storage.
    WEBCORE_EXPORT static Credential getFromPersistentStorage(const ProtectionSpace&);
#endif

    // These methods work for authentication schemes that support sending credentials without waiting for a request. E.g., for HTTP Basic authentication scheme
    // a client should assume that all paths at or deeper than the depth of a known protected resource share are within the same protection space.
    WEBCORE_EXPORT bool set(const String&, const Credential&, const URL&); // Returns true if the URL corresponds to a known protection space, so credentials could be updated.
    WEBCORE_EXPORT Credential get(const String&, const URL&);

    WEBCORE_EXPORT HashSet<SecurityOriginData> originsWithCredentials() const;

private:
    HashMap<std::pair<String /* partitionName */, ProtectionSpace>, Credential> m_protectionSpaceToCredentialMap;
    MemoryCompactRobinHoodHashSet<String> m_originsWithCredentials;

    typedef HashMap<String, ProtectionSpace> PathToDefaultProtectionSpaceMap;
    PathToDefaultProtectionSpaceMap m_pathToDefaultProtectionSpaceMap;

    PathToDefaultProtectionSpaceMap::iterator findDefaultProtectionSpaceForURL(const URL&);
};

} // namespace WebCore
