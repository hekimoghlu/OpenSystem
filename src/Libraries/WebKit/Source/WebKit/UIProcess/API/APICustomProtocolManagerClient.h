/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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

#include "LegacyCustomProtocolID.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
class LegacyCustomProtocolManagerProxy;
}

namespace WebCore {
class ResourceRequest;
}

namespace API {

class CustomProtocolManagerClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(CustomProtocolManagerClient);
public:
    virtual ~CustomProtocolManagerClient() { }

    virtual void startLoading(WebKit::LegacyCustomProtocolManagerProxy&, WebKit::LegacyCustomProtocolID, const WebCore::ResourceRequest&) { }
    virtual void stopLoading(WebKit::LegacyCustomProtocolManagerProxy&, WebKit::LegacyCustomProtocolID) { }

    virtual void invalidate(WebKit::LegacyCustomProtocolManagerProxy&) { }
};

} // namespace API
