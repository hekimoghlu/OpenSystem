/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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

#include "ProtectionSpaceBase.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS NSURLProtectionSpace;

namespace WebCore {

class ProtectionSpace : public ProtectionSpaceBase {
public:
    struct PlatformData {
        RetainPtr<NSURLProtectionSpace> nsSpace;
    };

    ProtectionSpace() : ProtectionSpaceBase() { }
    WEBCORE_EXPORT ProtectionSpace(const String& host, int port, ServerType, const String& realm, AuthenticationScheme, std::optional<PlatformData> = std::nullopt);
    ProtectionSpace(WTF::HashTableDeletedValueType deletedValue) : ProtectionSpaceBase(deletedValue) { }

    WEBCORE_EXPORT explicit ProtectionSpace(NSURLProtectionSpace *);

    static bool platformCompare(const ProtectionSpace& a, const ProtectionSpace& b);

    bool encodingRequiresPlatformData() const { return m_nsSpace && encodingRequiresPlatformData(m_nsSpace.get()); }

    WEBCORE_EXPORT bool receivesCredentialSecurely() const;
    WEBCORE_EXPORT NSURLProtectionSpace *nsSpace() const;
    
    WEBCORE_EXPORT std::optional<PlatformData> getPlatformDataToSerialize() const;

private:
    WEBCORE_EXPORT static bool encodingRequiresPlatformData(NSURLProtectionSpace *);

    mutable RetainPtr<NSURLProtectionSpace> m_nsSpace;
};

} // namespace WebCore
