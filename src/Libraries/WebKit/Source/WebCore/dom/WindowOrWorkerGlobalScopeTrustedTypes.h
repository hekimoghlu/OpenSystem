/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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

#include "Supplementable.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class TrustedTypePolicyFactory;
class DOMWindow;
class WeakPtrImplWithEventTargetData;
class WorkerGlobalScope;

template<typename> class ExceptionOr;

class WindowOrWorkerGlobalScopeTrustedTypes {
public:
    static ASCIILiteral workerGlobalSupplementName();
    static TrustedTypePolicyFactory* trustedTypes(DOMWindow&);
    static TrustedTypePolicyFactory* trustedTypes(WorkerGlobalScope&);
};

class WorkerGlobalScopeTrustedTypes : public Supplement<WorkerGlobalScope> {
    WTF_MAKE_TZONE_ALLOCATED(WorkerGlobalScopeTrustedTypes);
public:
    explicit WorkerGlobalScopeTrustedTypes(WorkerGlobalScope&);
    virtual ~WorkerGlobalScopeTrustedTypes();

    static WorkerGlobalScopeTrustedTypes* from(WorkerGlobalScope&);
    TrustedTypePolicyFactory* trustedTypes() const;

    void prepareForDestruction();

    static ASCIILiteral supplementName() { return WindowOrWorkerGlobalScopeTrustedTypes::workerGlobalSupplementName(); }

private:
    WeakPtr<WorkerGlobalScope, WeakPtrImplWithEventTargetData> m_scope;
    mutable RefPtr<TrustedTypePolicyFactory> m_trustedTypes;
};


} // namespace WebCore
