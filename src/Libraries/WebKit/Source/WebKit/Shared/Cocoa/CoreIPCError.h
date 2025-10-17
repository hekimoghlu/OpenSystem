/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

#import <CoreFoundation/CoreFoundation.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/text/WTFString.h>

OBJC_CLASS NSError;

namespace WebKit {

class CoreIPCError {
    WTF_MAKE_TZONE_ALLOCATED(CoreIPCError);
public:
    static bool hasValidUserInfo(const RetainPtr<CFDictionaryRef>&);

    CoreIPCError(CoreIPCError&&) = default;
    CoreIPCError& operator=(CoreIPCError&&) = default;

    CoreIPCError(NSError *);
    CoreIPCError(String&& domain, int64_t code, RetainPtr<CFDictionaryRef>&& userInfo, std::unique_ptr<CoreIPCError>&& underlyingError)
        : m_domain(WTFMove(domain))
        , m_code(WTFMove(code))
        , m_userInfo(WTFMove(userInfo))
        , m_underlyingError(WTFMove(underlyingError))
    {
    }

    RetainPtr<id> toID() const;

    String domain() const
    {
        return m_domain;
    }

    int64_t code() const
    {
        return m_code;
    }

    RetainPtr<CFDictionaryRef> userInfo() const
    {
        return m_userInfo;
    }

    const std::unique_ptr<CoreIPCError>& underlyingError() const
    {
        return m_underlyingError;
    }

private:
    bool isSafeToEncodeUserInfo(id value);

    String m_domain;
    int64_t m_code;
    RetainPtr<CFDictionaryRef> m_userInfo;
    std::unique_ptr<CoreIPCError> m_underlyingError;
};

}

#endif // PLATFORM(COCOA)
