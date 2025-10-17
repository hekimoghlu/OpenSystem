/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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

#include "CSSPaintCallback.h"
#include "WorkletGlobalScope.h"
#include <JavaScriptCore/JSObject.h>
#include <JavaScriptCore/Strong.h>
#include <wtf/Lock.h>

namespace WebCore {
struct PaintDefinition;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PaintDefinition> : std::true_type { };
}

namespace JSC {
class JSObject;
class VM;
} // namespace JSC

namespace WebCore {
class JSDOMGlobalObject;

// All paint definitions must be destroyed before the vm is destroyed, because otherwise they will point to freed memory.
struct PaintDefinition : public CanMakeWeakPtr<PaintDefinition> {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    PaintDefinition(const AtomString& name, JSC::JSObject* paintConstructor, Ref<CSSPaintCallback>&&, Vector<AtomString>&& inputProperties, Vector<String>&& inputArguments);

    const AtomString name;
    const JSC::JSObject* const paintConstructor;
    const Ref<CSSPaintCallback> paintCallback;
    const Vector<AtomString> inputProperties;
    const Vector<String> inputArguments;
};

class PaintWorkletGlobalScope final : public WorkletGlobalScope {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PaintWorkletGlobalScope);
public:
    static RefPtr<PaintWorkletGlobalScope> tryCreate(Document&, ScriptSourceCode&&);

    ExceptionOr<void> registerPaint(JSC::JSGlobalObject&, const AtomString& name, JSC::Strong<JSC::JSObject> paintConstructor);
    double devicePixelRatio() const;

    HashMap<String, std::unique_ptr<PaintDefinition>>& paintDefinitionMap() WTF_REQUIRES_LOCK(m_paintDefinitionLock);
    Lock& paintDefinitionLock() WTF_RETURNS_LOCK(m_paintDefinitionLock) { return m_paintDefinitionLock; }

    void prepareForDestruction() final
    {
        if (m_hasPreparedForDestruction)
            return;
        m_hasPreparedForDestruction = true;

        {
            Locker locker { paintDefinitionLock() };
            paintDefinitionMap().clear();
        }
        WorkletGlobalScope::prepareForDestruction();
    }

    FetchOptions::Destination destination() const final { return FetchOptions::Destination::Paintworklet; }

private:
    PaintWorkletGlobalScope(Document&, Ref<JSC::VM>&&, ScriptSourceCode&&);

    ~PaintWorkletGlobalScope()
    {
#if ASSERT_ENABLED
        Locker locker { paintDefinitionLock() };
        ASSERT(paintDefinitionMap().isEmpty());
#endif
    }

    bool isPaintWorkletGlobalScope() const final { return true; }

    HashMap<String, std::unique_ptr<PaintDefinition>> m_paintDefinitionMap WTF_GUARDED_BY_LOCK(m_paintDefinitionLock);
    Lock m_paintDefinitionLock;
    bool m_hasPreparedForDestruction { false };
};

inline auto PaintWorkletGlobalScope::paintDefinitionMap() -> HashMap<String, std::unique_ptr<PaintDefinition>>&
{
    ASSERT(m_paintDefinitionLock.isLocked());
    return m_paintDefinitionMap;
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PaintWorkletGlobalScope)
static bool isType(const WebCore::ScriptExecutionContext& context)
{
    auto* global = dynamicDowncast<WebCore::WorkletGlobalScope>(context);
    return global && global->isPaintWorkletGlobalScope();
}
static bool isType(const WebCore::WorkletGlobalScope& context) { return context.isPaintWorkletGlobalScope(); }
SPECIALIZE_TYPE_TRAITS_END()
