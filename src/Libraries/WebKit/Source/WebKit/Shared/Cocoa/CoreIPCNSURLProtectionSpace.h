/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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

#if PLATFORM(COCOA) && HAVE(WK_SECURE_CODING_NSURLPROTECTIONSPACE)

#import "CoreIPCData.h"
#import "CoreIPCSecTrust.h"
#import "CoreIPCString.h"

#import <WebCore/ProtectionSpace.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>

OBJC_CLASS NSURLProtectionSpace;

namespace WebKit {

struct CoreIPCNSURLProtectionSpaceData {
    std::optional<WebKit::CoreIPCString> host;
    uint16_t port;
    WebCore::ProtectionSpace::ServerType type;
    std::optional<WebKit::CoreIPCString> realm;
    WebCore::ProtectionSpace::AuthenticationScheme scheme;
    std::optional<CoreIPCSecTrust> trust;
    std::optional<Vector<WebKit::CoreIPCData>> distnames;
};

class CoreIPCNSURLProtectionSpace {
    WTF_MAKE_TZONE_ALLOCATED(CoreIPCNSURLProtectionSpace);
public:
    CoreIPCNSURLProtectionSpace(NSURLProtectionSpace *);
    CoreIPCNSURLProtectionSpace(CoreIPCNSURLProtectionSpaceData&&);
    CoreIPCNSURLProtectionSpace(const RetainPtr<NSURLProtectionSpace>&);

    RetainPtr<id> toID() const;
private:
    friend struct IPC::ArgumentCoder<CoreIPCNSURLProtectionSpace, void>;
    CoreIPCNSURLProtectionSpaceData m_data;
};

} // namespace WebKit

#endif // PLATFORM(COCOA) && HAVE(WK_SECURE_CODING_NSURLPROTECTIONSPACE)
