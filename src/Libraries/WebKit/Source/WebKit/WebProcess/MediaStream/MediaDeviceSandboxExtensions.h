/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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

#if ENABLE(MEDIA_STREAM)

#include "SandboxExtension.h"
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>

namespace WebKit {

class MediaDeviceSandboxExtensions {
    WTF_MAKE_NONCOPYABLE(MediaDeviceSandboxExtensions);
public:
    MediaDeviceSandboxExtensions() = default;
    MediaDeviceSandboxExtensions(MediaDeviceSandboxExtensions&&) = default;
    MediaDeviceSandboxExtensions& operator=(MediaDeviceSandboxExtensions&&) = default;

    MediaDeviceSandboxExtensions(Vector<String> ids, Vector<SandboxExtension::Handle>&& handles, SandboxExtension::Handle&& machBootstrapHandle);

    std::pair<String, RefPtr<SandboxExtension>> operator[](size_t i);
    size_t size() const;

    Vector<String> takeIDs() { return std::exchange(m_ids, { }); }
    Vector<SandboxExtension::Handle> takeHandles() { return std::exchange(m_handles, { }); }
    SandboxExtensionHandle takeMachBootstrapHandle() { return std::exchange(m_machBootstrapHandle, { }); }

    RefPtr<SandboxExtension> machBootstrapExtension() { return SandboxExtension::create(WTFMove(m_machBootstrapHandle)); }

private:
    Vector<String> m_ids;
    Vector<SandboxExtension::Handle> m_handles;
    SandboxExtension::Handle m_machBootstrapHandle;
};

} // namespace WebKit

#endif // ENABLE(MEDIA_STREAM)
