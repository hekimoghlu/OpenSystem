/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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

#include "ActiveDOMObject.h"
#include "ContextDestructionObserver.h"
#include "ExceptionOr.h"
#include "JSDOMPromiseDeferredForward.h"
#include "ScriptWrappable.h"
#include "WorkletOptions.h"
#include <wtf/HashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class WorkletGlobalScopeProxy;
class WorkletPendingTasks;

class Worklet : public RefCountedAndCanMakeWeakPtr<Worklet>, public ScriptWrappable, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Worklet);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    virtual ~Worklet();

    virtual void addModule(const String& moduleURL, WorkletOptions&&, DOMPromiseDeferred<void>&&);

    void finishPendingTasks(WorkletPendingTasks&);
    Document* document();

    const Vector<Ref<WorkletGlobalScopeProxy>>& proxies() const { return m_proxies; }
    const String& identifier() const { return m_identifier; }

protected:
    explicit Worklet(Document&);

private:
    virtual Vector<Ref<WorkletGlobalScopeProxy>> createGlobalScopes() = 0;

    String m_identifier;
    Vector<Ref<WorkletGlobalScopeProxy>> m_proxies;
    HashSet<RefPtr<WorkletPendingTasks>> m_pendingTasksSet;
};

} // namespace WebCore
