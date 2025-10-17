/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 21, 2022.
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
#include "JSDOMGuardedObject.h"

#include "JSDOMGlobalObjectInlines.h"

namespace WebCore {
using namespace JSC;

DOMGuardedObject::DOMGuardedObject(JSDOMGlobalObject& globalObject, JSCell& guarded)
    : ActiveDOMCallback(globalObject.scriptExecutionContext())
    , m_guarded(&guarded)
    , m_globalObject(&globalObject)
{
    if (globalObject.vm().heap.mutatorShouldBeFenced()) {
        Locker locker { globalObject.gcLock() };
        globalObject.guardedObjects().add(this);
    } else
        globalObject.guardedObjects(NoLockingNecessary).add(this);
    globalObject.vm().writeBarrier(&globalObject, &guarded);
}

DOMGuardedObject::~DOMGuardedObject()
{
    clear();
}

void DOMGuardedObject::clear()
{
    ASSERT(!m_guarded || m_globalObject);
    removeFromGlobalObject();
    m_guarded.clear();
}

void DOMGuardedObject::removeFromGlobalObject()
{
    if (!m_globalObject)
        return;

    if (m_globalObject->vm().heap.mutatorShouldBeFenced()) {
        Locker locker { m_globalObject->gcLock() };
        m_globalObject->guardedObjects().remove(this);
    } else
        m_globalObject->guardedObjects(NoLockingNecessary).remove(this);

    m_globalObject.clear();
}

void DOMGuardedObject::contextDestroyed()
{
    ActiveDOMCallback::contextDestroyed();
    clear();
}

}
