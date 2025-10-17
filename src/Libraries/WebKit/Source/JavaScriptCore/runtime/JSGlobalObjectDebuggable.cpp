/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#include "config.h"
#include "JSGlobalObjectDebuggable.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "InspectorAgentBase.h"
#include "InspectorFrontendChannel.h"
#include "JSGlobalObject.h"
#include "JSGlobalObjectInspectorController.h"
#include "JSLock.h"
#include "RemoteInspector.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/threads/BinarySemaphore.h>

using namespace Inspector;

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(JSGlobalObjectDebuggable);

Ref<JSGlobalObjectDebuggable> JSGlobalObjectDebuggable::create(JSGlobalObject& globalObject)
{
    return adoptRef(*new JSGlobalObjectDebuggable(globalObject));
}

JSGlobalObjectDebuggable::JSGlobalObjectDebuggable(JSGlobalObject& globalObject)
    : m_globalObject(&globalObject)
{
}

String JSGlobalObjectDebuggable::name() const
{
    if (m_globalObject && !m_globalObject->name().isEmpty())
        return m_globalObject->name();
    return "JSContext"_s;
}

void JSGlobalObjectDebuggable::connect(FrontendChannel& frontendChannel, bool automaticInspection, bool immediatelyPause)
{
    if (!m_globalObject)
        return;

    JSLockHolder locker(&m_globalObject->vm());
    m_globalObject->inspectorController().connectFrontend(frontendChannel, automaticInspection, immediatelyPause);
}

void JSGlobalObjectDebuggable::disconnect(FrontendChannel& frontendChannel)
{
    if (!m_globalObject)
        return;

    JSLockHolder locker(&m_globalObject->vm());

    m_globalObject->inspectorController().disconnectFrontend(frontendChannel);
}

void JSGlobalObjectDebuggable::dispatchMessageFromRemote(String&& message)
{
    if (!m_globalObject)
        return;

    JSLockHolder locker(&m_globalObject->vm());

    m_globalObject->inspectorController().dispatchMessageFromFrontend(WTFMove(message));
}

void JSGlobalObjectDebuggable::pauseWaitingForAutomaticInspection()
{
    if (!m_globalObject)
        return;

    JSC::JSLock::DropAllLocks dropAllLocks(&m_globalObject->vm());
    RemoteInspectionTarget::pauseWaitingForAutomaticInspection();
}

void JSGlobalObjectDebuggable::globalObjectDestroyed()
{
    m_globalObject = nullptr;
}

} // namespace JSC

#endif // ENABLE(REMOTE_INSPECTOR)
