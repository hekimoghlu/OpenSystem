/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#include "WindowOrWorkerGlobalScopeTrustedTypes.h"

#include "Document.h"
#include "LocalDOMWindow.h"
#include "LocalDOMWindowProperty.h"
#include "LocalFrame.h"
#include "Page.h"
#include "TrustedTypePolicyFactory.h"
#include "WorkerGlobalScope.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerGlobalScopeTrustedTypes);

class DOMWindowTrustedTypes : public Supplement<LocalDOMWindow>, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_ALLOCATED(DOMWindowTrustedTypes);
public:
    explicit DOMWindowTrustedTypes(LocalDOMWindow&);
    virtual ~DOMWindowTrustedTypes() = default;

    static DOMWindowTrustedTypes* from(LocalDOMWindow&);
    TrustedTypePolicyFactory* trustedTypes() const;

private:
    static WTF::ASCIILiteral supplementName() { return "DOMWindowTrustedTypes"_s; }

    mutable RefPtr<TrustedTypePolicyFactory> m_trustedTypes;
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(DOMWindowTrustedTypes);

DOMWindowTrustedTypes::DOMWindowTrustedTypes(LocalDOMWindow& window)
    : LocalDOMWindowProperty(&window)
{
}

DOMWindowTrustedTypes* DOMWindowTrustedTypes::from(LocalDOMWindow& window)
{
    auto* supplement = static_cast<DOMWindowTrustedTypes*>(Supplement<LocalDOMWindow>::from(&window, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<DOMWindowTrustedTypes>(window);
        supplement = newSupplement.get();
        provideTo(&window, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

TrustedTypePolicyFactory* DOMWindowTrustedTypes::trustedTypes() const
{
    if (!m_trustedTypes)
        m_trustedTypes = TrustedTypePolicyFactory::create(*window()->document());
    return m_trustedTypes.get();
}

TrustedTypePolicyFactory* WindowOrWorkerGlobalScopeTrustedTypes::trustedTypes(DOMWindow& window)
{
    RefPtr localWindow = dynamicDowncast<LocalDOMWindow>(window);
    if (!localWindow)
        return nullptr;
    return DOMWindowTrustedTypes::from(*localWindow)->trustedTypes();
}

WorkerGlobalScopeTrustedTypes::WorkerGlobalScopeTrustedTypes(WorkerGlobalScope& scope)
    : m_scope(scope)
{
}

WorkerGlobalScopeTrustedTypes* WorkerGlobalScopeTrustedTypes::from(WorkerGlobalScope& scope)
{
    auto* supplement = static_cast<WorkerGlobalScopeTrustedTypes*>(Supplement<WorkerGlobalScope>::from(&scope, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<WorkerGlobalScopeTrustedTypes>(scope);
        supplement = newSupplement.get();
        provideTo(&scope, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

WorkerGlobalScopeTrustedTypes::~WorkerGlobalScopeTrustedTypes() = default;

void WorkerGlobalScopeTrustedTypes::prepareForDestruction()
{
    m_trustedTypes = nullptr;
    m_scope = nullptr;
}

TrustedTypePolicyFactory* WorkerGlobalScopeTrustedTypes::trustedTypes() const
{
    if (!m_trustedTypes && m_scope)
        m_trustedTypes = TrustedTypePolicyFactory::create(Ref { *m_scope });
    return m_trustedTypes.get();
}

ASCIILiteral WindowOrWorkerGlobalScopeTrustedTypes::workerGlobalSupplementName()
{
    return "WorkerGlobalScopeTrustedTypes"_s;
}

TrustedTypePolicyFactory* WindowOrWorkerGlobalScopeTrustedTypes::trustedTypes(WorkerGlobalScope& scope)
{
    return WorkerGlobalScopeTrustedTypes::from(scope)->trustedTypes();
}

} // namespace WebCore
