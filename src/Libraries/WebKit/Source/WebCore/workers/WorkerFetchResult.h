/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#include "ContentSecurityPolicyResponseHeaders.h"
#include "CrossOriginEmbedderPolicy.h"
#include "ResourceError.h"
#include "ScriptBuffer.h"

namespace WebCore {

struct WorkerFetchResult {
    ScriptBuffer script;
    URL responseURL;
    CertificateInfo certificateInfo;
    ContentSecurityPolicyResponseHeaders contentSecurityPolicy;
    CrossOriginEmbedderPolicy crossOriginEmbedderPolicy;
    String referrerPolicy;
    ResourceError error;

    WorkerFetchResult isolatedCopy() const { return { script.isolatedCopy(), responseURL.isolatedCopy(), certificateInfo.isolatedCopy(), contentSecurityPolicy.isolatedCopy(), crossOriginEmbedderPolicy.isolatedCopy(), referrerPolicy.isolatedCopy(), error.isolatedCopy() }; }
};

inline WorkerFetchResult workerFetchError(const ResourceError& error)
{
    return { { }, { }, { }, { }, { }, { }, error };
}
} // namespace WebCore
