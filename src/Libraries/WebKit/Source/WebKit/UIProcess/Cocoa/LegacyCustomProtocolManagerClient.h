/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#import "APICustomProtocolManagerClient.h"
#import "LegacyCustomProtocolID.h"
#import <wtf/HashMap.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>

OBJC_CLASS WKCustomProtocolLoader;

namespace WebCore {
class ResourceRequest;
}

namespace WebKit {

class LegacyCustomProtocolManagerProxy;

class LegacyCustomProtocolManagerClient final : public API::CustomProtocolManagerClient {
    WTF_MAKE_TZONE_ALLOCATED(LegacyCustomProtocolManagerClient);
public:
    void startLoading(LegacyCustomProtocolManagerProxy&, LegacyCustomProtocolID, const WebCore::ResourceRequest&) final;
    void stopLoading(LegacyCustomProtocolManagerProxy&, LegacyCustomProtocolID) final;
    void invalidate(LegacyCustomProtocolManagerProxy&) final;
private:
    HashMap<LegacyCustomProtocolID, RetainPtr<WKCustomProtocolLoader>> m_loaderMap;
};

} // namespace WebKit

