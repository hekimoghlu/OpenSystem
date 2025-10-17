/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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

#include "JSRemoteInspector.h"
#include "RemoteControllableTarget.h"
#include <wtf/ProcessID.h>
#include <wtf/RetainPtr.h>
#include <wtf/TypeCasts.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

class FrontendChannel;

class RemoteInspectionTarget : public RemoteControllableTarget {
public:
    JS_EXPORT_PRIVATE RemoteInspectionTarget();
    JS_EXPORT_PRIVATE ~RemoteInspectionTarget() override;
    JS_EXPORT_PRIVATE bool inspectable() const;
    JS_EXPORT_PRIVATE void setInspectable(bool);

    bool allowsInspectionByPolicy() const;

#if USE(CF)
    CFRunLoopRef targetRunLoop() const final { return m_runLoop.get(); }
    void setTargetRunLoop(CFRunLoopRef runLoop) { m_runLoop = runLoop; }
#endif

    virtual String name() const { return String(); } // ITML JavaScript Page ServiceWorker WebPage
    virtual String url() const { return String(); } // Page ServiceWorker WebPage
    virtual const String& nameOverride() const { return nullString(); }
    virtual bool hasLocalDebugger() const = 0;

    virtual void setIndicating(bool) { } // Default is to do nothing.

    virtual bool automaticInspectionAllowed() const { return false; }
    JS_EXPORT_PRIVATE virtual void pauseWaitingForAutomaticInspection();
    JS_EXPORT_PRIVATE virtual void unpauseForInitializedInspector();

    // RemoteControllableTarget overrides.
    JS_EXPORT_PRIVATE bool remoteControlAllowed() const final;

    std::optional<ProcessID> presentingApplicationPID() const { return m_presentingApplicationPID; }
    JS_EXPORT_PRIVATE void setPresentingApplicationPID(std::optional<ProcessID>&&);

private:
    enum class Inspectable : uint8_t {
        Yes,
        No,

        // For WebKit internal proxies and wrappers, we want to always disable inspection even when internal policies
        // would otherwise enable inspection.
        NoIgnoringInternalPolicies,
    };
    Inspectable m_inspectable { JSRemoteInspectorGetInspectionFollowsInternalPolicies() ? Inspectable::No : Inspectable::NoIgnoringInternalPolicies };

#if USE(CF)
    RetainPtr<CFRunLoopRef> m_runLoop;
#endif

    std::optional<ProcessID> m_presentingApplicationPID;
};

} // namespace Inspector

SPECIALIZE_TYPE_TRAITS_BEGIN(Inspector::RemoteInspectionTarget)
    static bool isType(const Inspector::RemoteControllableTarget& target)
    {
        return target.type() != Inspector::RemoteControllableTarget::Type::Automation;
    }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(REMOTE_INSPECTOR)
