/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

#if ENABLE(BUBBLEWRAP_SANDBOX)
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>
#include <wtf/unix/UnixFileDescriptor.h>

namespace WebKit {

struct ProcessLaunchOptions;

class XDGDBusProxy {
    WTF_MAKE_TZONE_ALLOCATED(XDGDBusProxy);
    WTF_MAKE_NONCOPYABLE(XDGDBusProxy);
public:
    XDGDBusProxy() = default;
    ~XDGDBusProxy() = default;

    enum class AllowPortals : bool { No, Yes };
    std::optional<CString> dbusSessionProxy(const char* baseDirectory, AllowPortals);
#if USE(ATSPI)
    std::optional<CString> accessibilityProxy(const char* baseDirectory, const String& sandboxedAccessibilityBusPath, const String& accessibilityBusName);
#endif

    void launch(const ProcessLaunchOptions&);

private:
    static CString makeProxy(const char* baseDirectory, const char* proxyTemplate);

    Vector<CString> m_args;
    CString m_dbusSessionProxyPath;
#if USE(ATSPI)
    CString m_accessibilityProxyPath;
#endif
    UnixFileDescriptor m_syncFD;
};

} // namespace WebKit

#endif // ENABLE(BUBBLEWRAP_SANDBOX)
