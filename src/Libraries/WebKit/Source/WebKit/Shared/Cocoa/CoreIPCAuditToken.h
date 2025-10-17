/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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

#include <array>
#include <mach/mach.h>

#include <wtf/StdLibExtras.h>

namespace WebKit {

static std::array<unsigned, 8> invalidAuditToken()
{
    static std::array<unsigned, 8> invalidAuditToken;
    invalidAuditToken.fill(std::numeric_limits<unsigned>::max());
    return invalidAuditToken;
}

struct CoreIPCAuditToken {
    CoreIPCAuditToken()
        : CoreIPCAuditToken { invalidAuditToken() }
    {
    }

    CoreIPCAuditToken(audit_token_t input)
    {
        memcpySpan(asMutableByteSpan(token), asByteSpan(input));
    }

    CoreIPCAuditToken(std::array<unsigned, 8> token)
        : token { WTFMove(token) }
    {
    }

    audit_token_t auditToken() const
    {
        audit_token_t result;
        memcpySpan(asMutableByteSpan(result), asByteSpan(token));
        return result;
    }

    std::array<unsigned, 8> token;
    static_assert(sizeof(token) == sizeof(audit_token_t));
};

} // namespace WebKit
