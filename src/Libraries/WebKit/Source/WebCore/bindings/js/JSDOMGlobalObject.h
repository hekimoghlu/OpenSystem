/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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

#include <JavaScriptCore/JSGlobalObject.h>
#include <JavaScriptCore/WeakGCMap.h>
#include <wtf/Compiler.h>
#include <wtf/Forward.h>

namespace JSC {

enum class JSPromiseRejectionOperation : unsigned;

}

namespace WebCore {

class DOMConstructors;
class DOMGuardedObject;
class JSBuiltinInternalFunctions;
class Event;
class DOMWrapperWorld;
class ScriptExecutionContext;

using JSDOMStructureMap = UncheckedKeyHashMap<const JSC::ClassInfo*, JSC::WriteBarrier<JSC::Structure>>;
using DOMGuardedObjectSet = UncheckedKeyHashSet<DOMGuardedObject*>;

class WEBCORE_EXPORT JSDOMGlobalObject : public JSC::JSGlobalObject {
public:
    struct JSDOMGlobalObjectData;

    using Base = JSC::JSGlobalObject;

    static const JSC::ClassInfo s_info;

    template<typename, JSC::SubspaceAccess>
    static void subspaceFor(JSC::VM&) { RELEASE_ASSERT_NOT_REACHED(); }

    static void destroy(JSC::JSCell*);

public:
    Lock& gcLock() WTF_RETURNS_LOCK(m_gcLock) { return m_gcLock; }

    JSDOMStructureMap& structures() WTF_REQUIRES_LOCK(m_gcLock) { return m_structures; }
    DOMGuardedObjectSet& guardedObjects() WTF_REQUIRES_LOCK(m_gcLock) { return m_guardedObjects; }
    DOMConstructors& constructors() { return *m_constructors; }

    // No locking is necessary for call sites that do not mutate the containers and are not on the GC thread.
    const JSDOMStructureMap& structures() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(!Thread::mayBeGCThread()); return m_structures; }
    const DOMGuardedObjectSet& guardedObjects() const WTF_IGNORES_THREAD_SAFETY_ANALYSIS { ASSERT(!Thread::mayBeGCThread()); return m_guardedObjects; }
    const DOMConstructors& constructors() const { ASSERT(!Thread::mayBeGCThread()); return *m_constructors; }

    // The following don't require grabbing the gcLock first and should only be called when the Heap says that mutators don't have to be fenced.
    inline JSDOMStructureMap& structures(NoLockingNecessaryTag);
    inline DOMGuardedObjectSet& guardedObjects(NoLockingNecessaryTag);

    ScriptExecutionContext* scriptExecutionContext() const;

    static String codeForEval(JSC::JSGlobalObject*, JSC::JSValue);
    static bool canCompileStrings(JSC::JSGlobalObject*, JSC::CompilationType, String, const JSC::ArgList&);
    static JSC::Structure* trustedScriptStructure(JSC::JSGlobalObject*);

    // https://tc39.es/ecma262/#sec-agent-clusters
    String agentClusterID() const;
    static String defaultAgentClusterID();

    // Make binding code generation easier.
    JSDOMGlobalObject* globalObject() { return this; }

    DECLARE_VISIT_CHILDREN;

    DOMWrapperWorld& world() { return m_world.get(); }
    bool worldIsNormal() const { return m_worldIsNormal; }
    static constexpr ptrdiff_t offsetOfWorldIsNormal() { return OBJECT_OFFSETOF(JSDOMGlobalObject, m_worldIsNormal); }

    JSBuiltinInternalFunctions& builtinInternalFunctions() { return m_builtinInternalFunctions; }

    static void reportUncaughtExceptionAtEventLoop(JSGlobalObject*, JSC::Exception*);
    static JSC::JSGlobalObject* deriveShadowRealmGlobalObject(JSC::JSGlobalObject*);

    void clearDOMGuardedObjects() const;

    JSC::JSGlobalProxy& proxy() const { ASSERT(m_proxy); return *m_proxy.get(); }

    JSC::JSFunction* createCrossOriginFunction(JSC::JSGlobalObject*, JSC::PropertyName, JSC::NativeFunction, unsigned length);
    JSC::GetterSetter* createCrossOriginGetterSetter(JSC::JSGlobalObject*, JSC::PropertyName, JSC::GetValueFunc, JSC::PutValueFunc);

public:
    ~JSDOMGlobalObject();

    static constexpr const JSC::ClassInfo* info() { return &s_info; }

    inline static JSC::Structure* createStructure(JSC::VM&, JSC::JSValue);

protected:
    JSDOMGlobalObject(JSC::VM&, JSC::Structure*, Ref<DOMWrapperWorld>&&, const JSC::GlobalObjectMethodTable* = nullptr);
    void finishCreation(JSC::VM&);
    void finishCreation(JSC::VM&, JSC::JSObject*);

    static void promiseRejectionTracker(JSC::JSGlobalObject*, JSC::JSPromise*, JSC::JSPromiseRejectionOperation);

#if ENABLE(WEBASSEMBLY)
    static JSC::JSPromise* compileStreaming(JSC::JSGlobalObject*, JSC::JSValue);
    static JSC::JSPromise* instantiateStreaming(JSC::JSGlobalObject*, JSC::JSValue, JSC::JSObject*);
#endif

    static JSC::Identifier moduleLoaderResolve(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue, JSC::JSValue, JSC::JSValue);
    static JSC::JSInternalPromise* moduleLoaderFetch(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue, JSC::JSValue, JSC::JSValue);
    static JSC::JSValue moduleLoaderEvaluate(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue, JSC::JSValue, JSC::JSValue, JSC::JSValue, JSC::JSValue);
    static JSC::JSInternalPromise* moduleLoaderImportModule(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSString*, JSC::JSValue, const JSC::SourceOrigin&);
    static JSC::JSObject* moduleLoaderCreateImportMetaProperties(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue, JSC::JSModuleRecord*, JSC::JSValue);

    JSDOMStructureMap m_structures WTF_GUARDED_BY_LOCK(m_gcLock);
    DOMGuardedObjectSet m_guardedObjects WTF_GUARDED_BY_LOCK(m_gcLock);
    std::unique_ptr<DOMConstructors> m_constructors;

    Ref<DOMWrapperWorld> m_world;
    uint8_t m_worldIsNormal;
    Lock m_gcLock;
    JSC::WriteBarrier<JSC::JSGlobalProxy> m_proxy;

private:
    void addBuiltinGlobals(JSC::VM&);
    friend JSBuiltinInternalFunctions;

    using CrossOriginMapKey = std::pair<JSC::JSGlobalObject*, void*>;

    UniqueRef<JSBuiltinInternalFunctions> m_builtinInternalFunctions;
    JSC::WeakGCMap<CrossOriginMapKey, JSC::JSFunction> m_crossOriginFunctionMap;
    JSC::WeakGCMap<CrossOriginMapKey, JSC::GetterSetter> m_crossOriginGetterSetterMap;
};

JSDOMGlobalObject* toJSDOMGlobalObject(ScriptExecutionContext&, DOMWrapperWorld&);
WEBCORE_EXPORT JSDOMGlobalObject& callerGlobalObject(JSC::JSGlobalObject&, JSC::CallFrame*);
JSDOMGlobalObject& legacyActiveGlobalObjectForAccessor(JSC::JSGlobalObject&, JSC::CallFrame*);

template<class JSClass>
inline JSClass* toJSDOMGlobalObject(JSC::VM&, JSC::JSValue);

} // namespace WebCore
