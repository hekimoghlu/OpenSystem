/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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

#include "CertificateInfo.h"
#include "ProtectionSpaceBase.h"

namespace WebCore {

class ProtectionSpace : public ProtectionSpaceBase {
public:
    ProtectionSpace()
        : ProtectionSpaceBase()
    {
    }

    ProtectionSpace(const String& host, int port, ServerType serverType, const String& realm, AuthenticationScheme authenticationScheme)
        : ProtectionSpaceBase(host, port, serverType, realm, authenticationScheme)
    {
    }

    ProtectionSpace(const String& host, int port, ServerType serverType, const String& realm, AuthenticationScheme authenticationScheme, const CertificateInfo& certificateInfo)
        : ProtectionSpaceBase(host, port, serverType, realm, authenticationScheme)
        , m_certificateInfo(certificateInfo)
    {
    }

    ProtectionSpace(WTF::HashTableDeletedValueType deletedValue)
        : ProtectionSpaceBase(deletedValue)
    {
    }

    bool encodingRequiresPlatformData() const { return true; }
    static bool platformCompare(const ProtectionSpace& a, const ProtectionSpace& b) { return a.m_certificateInfo == b.m_certificateInfo; }

    WEBCORE_EXPORT const CertificateInfo& certificateInfo() const;

private:

    CertificateInfo m_certificateInfo;
};

} // namespace WebCore
