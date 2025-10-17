/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#ifndef KJS_BINDINGS_OBJC_RUNTIME_H
#define KJS_BINDINGS_OBJC_RUNTIME_H

#include "BridgeJSC.h"
#include "JSDOMBinding.h"
#include "objc_header.h"
#include <JavaScriptCore/JSGlobalObject.h>
#include <wtf/RetainPtr.h>

namespace JSC {
namespace Bindings {

ClassStructPtr webScriptObjectClass();
ClassStructPtr webUndefinedClass();

class ObjcInstance;

class ObjcField : public Field {
public:
    ObjcField(IvarStructPtr);
    ObjcField(CFStringRef name);
    
    virtual JSValue valueFromInstance(JSGlobalObject*, const Instance*) const;
    virtual bool setValueToInstance(JSGlobalObject*, const Instance*, JSValue) const;

private:
    IvarStructPtr _ivar;
    RetainPtr<CFStringRef> _name;
};

class ObjcMethod : public Method {
public:
    ObjcMethod() : _objcClass(0), _selector(0), _javaScriptName(0) {}
    ObjcMethod(ClassStructPtr aClass, SEL _selector);

    virtual int numParameters() const;

    NSMethodSignature *getMethodSignature() const;

    bool isFallbackMethod() const;
    void setJavaScriptName(CFStringRef n) { _javaScriptName = n; }
    CFStringRef javaScriptName() const { return _javaScriptName.get(); }
    
    SEL selector() const { return _selector; }

private:
    ClassStructPtr _objcClass;
    SEL _selector;
    RetainPtr<CFStringRef> _javaScriptName;
};

class ObjcArray : public Array {
public:
    ObjcArray(ObjectStructPtr, RefPtr<RootObject>&&);

    virtual bool setValueAt(JSGlobalObject*, unsigned int index, JSValue aValue) const;
    virtual JSValue valueAt(JSGlobalObject*, unsigned int index) const;
    virtual unsigned int getLength() const;
    
    ObjectStructPtr getObjcArray() const { return _array.get(); }

private:
    RetainPtr<ObjectStructPtr> _array;
};

class ObjcFallbackObjectImp final : public JSDestructibleObject {
public:
    using Base = JSDestructibleObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetCallData | OverridesPut;
    static constexpr JSC::DestructionMode needsDestruction = JSC::NeedsDestruction;

    template<typename CellType, JSC::SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        return subspaceForImpl(vm);
    }

    static ObjcFallbackObjectImp* create(JSGlobalObject* exec, JSGlobalObject* globalObject, ObjcInstance* instance, const String& propertyName)
    {
        VM& vm = globalObject->vm();
        // FIXME: deprecatedGetDOMStructure uses the prototype off of the wrong global object
        Structure* domStructure = WebCore::deprecatedGetDOMStructure<ObjcFallbackObjectImp>(exec);
        ObjcFallbackObjectImp* fallbackObject = new (NotNull, allocateCell<ObjcFallbackObjectImp>(vm)) ObjcFallbackObjectImp(globalObject, domStructure, instance, propertyName);
        fallbackObject->finishCreation(globalObject);
        return fallbackObject;
    }

    DECLARE_INFO;

    const String& propertyName() const { return m_item; }

    static ObjectPrototype* createPrototype(VM&, JSGlobalObject& globalObject)
    {
        return globalObject.objectPrototype();
    }

    static Structure* createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
    {
        return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
    }

    ObjcInstance* getInternalObjCInstance() const { return _instance.get(); }

private:
    ObjcFallbackObjectImp(JSGlobalObject*, Structure*, ObjcInstance*, const String& propertyName);
    void finishCreation(JSGlobalObject*);

    static void destroy(JSCell*);
    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static CallData getCallData(JSCell*);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);

    bool toBoolean(JSGlobalObject*) const; // FIXME: Currently this is broken because none of the superclasses are marked virtual. We need to solve this in the longer term.

    static GCClient::IsoSubspace* subspaceForImpl(VM&);

    RefPtr<ObjcInstance> _instance;
    String m_item;
};

} // namespace Bindings
} // namespace JSC

#endif
