/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#ifndef RUNTIME_ROOT_H_
#define RUNTIME_ROOT_H_

#include <JavaScriptCore/Strong.h>
#include <JavaScriptCore/Weak.h>
#include <JavaScriptCore/WeakInlines.h>
#include <wtf/Forward.h>
#include <wtf/HashCountedSet.h>
#include <wtf/HashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>

namespace JSC {

class Interpreter;
class JSGlobalObject;

namespace Bindings {

class RootObject;
class RuntimeObject;

typedef HashCountedSet<JSObject*> ProtectCountSet;

extern RootObject* findProtectingRootObject(JSObject*);
extern RootObject* findRootObject(JSGlobalObject*);

class RootObject : public RefCounted<RootObject>, private JSC::WeakHandleOwner {
    friend class JavaJSObject;

public:
    WEBCORE_EXPORT virtual ~RootObject();
    
    static Ref<RootObject> create(const void* nativeHandle, JSGlobalObject*);

    bool isValid() { return m_isValid; }
    void invalidate();
    
    void gcProtect(JSObject*);
    void gcUnprotect(JSObject*);
    bool gcIsProtected(JSObject*);

    const void* nativeHandle() const;
    WEBCORE_EXPORT JSGlobalObject* globalObject() const;
    void updateGlobalObject(JSGlobalObject*);

    void addRuntimeObject(VM&, RuntimeObject*);
    void removeRuntimeObject(RuntimeObject*);

    struct InvalidationCallback {
        virtual void operator()(RootObject*) = 0;
        virtual ~InvalidationCallback();
    };
    void addInvalidationCallback(InvalidationCallback* callback) { m_invalidationCallbacks.add(callback); }

private:
    RootObject(const void* nativeHandle, JSGlobalObject*);

    // WeakHandleOwner
    void finalize(JSC::Handle<JSC::Unknown>, void* context) override;

    bool m_isValid;
    
    const void* m_nativeHandle;
    Strong<JSGlobalObject> m_globalObject;

    ProtectCountSet m_protectCountSet;
    HashMap<RuntimeObject*, JSC::Weak<RuntimeObject>> m_runtimeObjects; // We use a map to implement a set.

    UncheckedKeyHashSet<InvalidationCallback*> m_invalidationCallbacks;
};

} // namespace Bindings

} // namespace JSC

#endif // RUNTIME_ROOT_H_
