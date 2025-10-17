/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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

#include "CrossOriginEmbedderPolicy.h"
#include "ResourceRequest.h"
#include "SecurityOrigin.h"

namespace WebCore {

struct RetrieveRecordsOptions {
    RetrieveRecordsOptions isolatedCopy() const & { return { request.isolatedCopy(), crossOriginEmbedderPolicy.isolatedCopy(), sourceOrigin->isolatedCopy(), ignoreSearch, ignoreMethod, ignoreVary, shouldProvideResponse }; }
    RetrieveRecordsOptions isolatedCopy() && { return { WTFMove(request).isolatedCopy(), WTFMove(crossOriginEmbedderPolicy).isolatedCopy(), sourceOrigin->isolatedCopy(), ignoreSearch, ignoreMethod, ignoreVary, shouldProvideResponse }; }

    ResourceRequest request;
    CrossOriginEmbedderPolicy crossOriginEmbedderPolicy;
    Ref<SecurityOrigin> sourceOrigin;
    bool ignoreSearch { false };
    bool ignoreMethod { false };
    bool ignoreVary { false };
    bool shouldProvideResponse { true };
};

} // namespace WebCore
