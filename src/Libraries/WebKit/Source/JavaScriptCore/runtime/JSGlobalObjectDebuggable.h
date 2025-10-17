/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteInspectionTarget.h"
#include <wtf/Noncopyable.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace Inspector {
class FrontendChannel;
enum class DisconnectReason;
}

namespace JSC {

class JSGlobalObject;

class JSGlobalObjectDebuggable final : public Inspector::RemoteInspectionTarget {
    WTF_MAKE_TZONE_ALLOCATED(JSGlobalObjectDebuggable);
    WTF_MAKE_NONCOPYABLE(JSGlobalObjectDebuggable);
public:
    static Ref<JSGlobalObjectDebuggable> create(JSGlobalObject&);
    ~JSGlobalObjectDebuggable() final { }

    Inspector::RemoteControllableTarget::Type type() const final { return m_type; }
    void setIsITML() { m_type = Inspector::RemoteControllableTarget::Type::ITML; }

    String name() const final;
    bool hasLocalDebugger() const final { return false; }

    void connect(Inspector::FrontendChannel&, bool isAutomaticConnection = false, bool immediatelyPause = false) final;
    void disconnect(Inspector::FrontendChannel&) final;
    void dispatchMessageFromRemote(String&& message) final;

    bool automaticInspectionAllowed() const final { return true; }
    void pauseWaitingForAutomaticInspection() final;

    void globalObjectDestroyed();

private:
    JSGlobalObjectDebuggable(JSGlobalObject&);

    JSGlobalObject* m_globalObject;
    Inspector::RemoteControllableTarget::Type m_type { Inspector::RemoteControllableTarget::Type::JavaScript };
};

} // namespace JSC

SPECIALIZE_TYPE_TRAITS_BEGIN(JSC::JSGlobalObjectDebuggable)
    static bool isType(const Inspector::RemoteControllableTarget& target)
    {
        return target.type() == Inspector::RemoteControllableTarget::Type::JavaScript
            || target.type() == Inspector::RemoteControllableTarget::Type::ITML;
    }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(REMOTE_INSPECTOR)
