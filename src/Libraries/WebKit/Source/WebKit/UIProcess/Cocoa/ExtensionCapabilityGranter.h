/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

#if ENABLE(EXTENSION_CAPABILITIES)

#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/CheckedPtr.h>
#include <wtf/FastMalloc.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class ExtensionCapability;
class ExtensionCapabilityGrant;
class ExtensionCapabilityGranter;
class GPUProcessProxy;
class MediaCapability;
class WebPageProxy;
class WebProcessProxy;

struct ExtensionCapabilityGranterClient : public AbstractRefCountedAndCanMakeWeakPtr<ExtensionCapabilityGranterClient> {
    virtual ~ExtensionCapabilityGranterClient() = default;

    virtual RefPtr<GPUProcessProxy> gpuProcessForCapabilityGranter(const ExtensionCapabilityGranter&) = 0;
    virtual RefPtr<WebProcessProxy> webProcessForCapabilityGranter(const ExtensionCapabilityGranter&, const String& environmentIdentifier) = 0;
};

class ExtensionCapabilityGranter : public RefCountedAndCanMakeWeakPtr<ExtensionCapabilityGranter> {
    WTF_MAKE_TZONE_ALLOCATED(ExtensionCapabilityGranter);
    WTF_MAKE_NONCOPYABLE(ExtensionCapabilityGranter);
public:
    static RefPtr<ExtensionCapabilityGranter> create(ExtensionCapabilityGranterClient&);

    void grant(const ExtensionCapability&);
    void revoke(const ExtensionCapability&);

    void setMediaCapabilityActive(MediaCapability&, bool);
    void invalidateGrants(Vector<ExtensionCapabilityGrant>&&);

private:
    explicit ExtensionCapabilityGranter(ExtensionCapabilityGranterClient&);

    WeakRef<ExtensionCapabilityGranterClient> m_client;
};

} // namespace WebKit

#endif // ENABLE(EXTENSION_CAPABILITIES)
