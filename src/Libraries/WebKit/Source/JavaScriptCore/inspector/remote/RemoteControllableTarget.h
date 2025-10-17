/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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

#include "JSExportMacros.h"
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

#if USE(CF)
#include <CoreFoundation/CFRunLoop.h>
#endif

namespace Inspector {

class FrontendChannel;

using TargetID = unsigned;

class JS_EXPORT_PRIVATE RemoteControllableTarget : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteControllableTarget> {
public:
    RemoteControllableTarget();
    virtual ~RemoteControllableTarget();

    void init();
    void update();

    virtual void connect(FrontendChannel&, bool isAutomaticConnection = false, bool immediatelyPause = false) = 0;
    virtual void disconnect(FrontendChannel&) = 0;

    TargetID targetIdentifier() const { return m_identifier; }
    void setTargetIdentifier(TargetID identifier) { m_identifier = identifier; }

    enum class Type {
        Automation,
        ITML,
        JavaScript,
        Page,
        ServiceWorker,
        WebPage,
    };
    virtual Type type() const = 0;
    virtual bool remoteControlAllowed() const = 0;
    virtual void dispatchMessageFromRemote(String&& message) = 0;

#if USE(CF)
    // The dispatch block will be scheduled on a global run loop if null is returned.
    virtual CFRunLoopRef targetRunLoop() const { return nullptr; }
#endif

private:
    TargetID m_identifier { 0 };
};

} // namespace Inspector

#define SPECIALIZE_TYPE_TRAITS_CONTROLLABLE_TARGET(ToClassName, ToClassType) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToClassName) \
    static bool isType(const Inspector::RemoteControllableTarget& target) { return target.type() == Inspector::RemoteControllableTarget::Type::ToClassType; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(REMOTE_INSPECTOR)
