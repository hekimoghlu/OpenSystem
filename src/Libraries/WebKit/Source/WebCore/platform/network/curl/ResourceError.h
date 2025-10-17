/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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

#include "ResourceErrorBase.h"

namespace WebCore {

class ResourceError : public ResourceErrorBase {
public:
    ResourceError(Type type = Type::Null)
        : ResourceErrorBase(type)
    {
    }

    ResourceError(const String& domain, int errorCode, const URL& failingURL, const String& localizedDescription, Type type = Type::General, IsSanitized isSanitized = IsSanitized::No)
        : ResourceErrorBase(domain, errorCode, failingURL, localizedDescription, type, isSanitized)
    {
    }

    struct IPCData {
        Type type;
        String domain;
        int errorCode;
        URL failingURL;
        String localizedDescription;
        IsSanitized isSanitized;
    };
    WEBCORE_EXPORT static ResourceError fromIPCData(std::optional<IPCData>&&);
    WEBCORE_EXPORT std::optional<IPCData> ipcData() const;

    WEBCORE_EXPORT ResourceError(int curlCode, const URL& failingURL, Type = Type::General);

    WEBCORE_EXPORT bool isCertificationVerificationError() const;

    ErrorRecoveryMethod errorRecoveryMethod() const { return ErrorRecoveryMethod::NoRecovery; }

    static bool platformCompare(const ResourceError& a, const ResourceError& b);

private:
    friend class ResourceErrorBase;

    void doPlatformIsolatedCopy(const ResourceError&);
};

} // namespace WebCore
