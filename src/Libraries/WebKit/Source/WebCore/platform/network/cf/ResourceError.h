/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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

#include <wtf/RetainPtr.h>

OBJC_CLASS NSError;

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
        ASSERT(domain != getNSURLErrorDomain());
        ASSERT(domain != getCFErrorDomainCFNetwork());
    }

    WEBCORE_EXPORT ResourceError(CFErrorRef error);

    WEBCORE_EXPORT CFErrorRef cfError() const;
    WEBCORE_EXPORT operator CFErrorRef() const;
    WEBCORE_EXPORT ResourceError(NSError *);
    WEBCORE_EXPORT NSError *nsError() const;
    WEBCORE_EXPORT operator NSError *() const;


    struct IPCData {
        Type type;
        RetainPtr<NSError> nsError;
        bool isSanitized;
    };
    WEBCORE_EXPORT static ResourceError fromIPCData(std::optional<IPCData>&&);
    WEBCORE_EXPORT std::optional<IPCData> ipcData() const;

#if ENABLE(ADVANCED_PRIVACY_PROTECTIONS)
    WEBCORE_EXPORT bool blockedKnownTracker() const;
    WEBCORE_EXPORT String blockedTrackerHostName() const;
#endif

    WEBCORE_EXPORT ErrorRecoveryMethod errorRecoveryMethod() const;

    WEBCORE_EXPORT bool hasMatchingFailingURLKeys() const;

    static bool platformCompare(const ResourceError& a, const ResourceError& b);


private:
    friend class ResourceErrorBase;

    WEBCORE_EXPORT const String& getNSURLErrorDomain() const;
    WEBCORE_EXPORT const String& getCFErrorDomainCFNetwork() const;
    WEBCORE_EXPORT void mapPlatformError();

    void platformLazyInit();

    void doPlatformIsolatedCopy(const ResourceError&);

    mutable RetainPtr<NSError> m_platformError;
    bool m_dataIsUpToDate { true };
};

} // namespace WebCore
