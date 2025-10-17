/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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

#include <JavaScriptCore/JSString.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC  {

class ArgList;
class Identifier;
class JSGlobalObject;
class PropertyNameArray;
class RuntimeMethod;

namespace Bindings {

class Instance;
class Method;
class RootObject;
class RuntimeObject;

class Field {
    WTF_MAKE_TZONE_ALLOCATED(Field);
public:
    virtual JSValue valueFromInstance(JSGlobalObject*, const Instance*) const = 0;
    virtual bool setValueToInstance(JSGlobalObject*, const Instance*, JSValue) const = 0;

    virtual ~Field() = default;
};

class Method {
    WTF_MAKE_TZONE_ALLOCATED(Method);
    WTF_MAKE_NONCOPYABLE(Method);
public:
    Method() = default;
    virtual int numParameters() const = 0;

    virtual ~Method() = default;
};

class Class {
    WTF_MAKE_TZONE_ALLOCATED(Class);
    WTF_MAKE_NONCOPYABLE(Class);
public:
    Class() = default;
    virtual Method* methodNamed(PropertyName, Instance*) const = 0;
    virtual Field* fieldNamed(PropertyName, Instance*) const = 0;
    virtual JSValue fallbackObject(JSGlobalObject*, Instance*, PropertyName) { return jsUndefined(); }

    virtual ~Class() = default;
};

class Instance : public RefCounted<Instance> {
public:
    WEBCORE_EXPORT Instance(RefPtr<RootObject>&&);

    // These functions are called before and after the main entry points into
    // the native implementations.  They can be used to establish and cleanup
    // any needed state.
    void begin();
    void end();

    virtual Class* getClass() const = 0;
    WEBCORE_EXPORT JSObject* createRuntimeObject(JSGlobalObject*);
    void willInvalidateRuntimeObject();

    // Returns false if the value was not set successfully.
    virtual bool setValueOfUndefinedField(JSGlobalObject*, PropertyName, JSValue) { return false; }

    virtual JSValue getMethod(JSGlobalObject*, PropertyName) = 0;
    virtual JSValue invokeMethod(JSGlobalObject*, CallFrame*, RuntimeMethod* method) = 0;

    virtual bool supportsInvokeDefaultMethod() const { return false; }
    virtual JSValue invokeDefaultMethod(JSGlobalObject*, CallFrame*) { return jsUndefined(); }

    virtual bool supportsConstruct() const { return false; }
    virtual JSValue invokeConstruct(JSGlobalObject*, CallFrame*, const ArgList&) { return JSValue(); }

    virtual void getPropertyNames(JSGlobalObject*, PropertyNameArray&) { }

    virtual JSValue defaultValue(JSGlobalObject*, PreferredPrimitiveType) const = 0;

    virtual JSValue valueOf(JSGlobalObject* exec) const = 0;

    RootObject* rootObject() const;

    WEBCORE_EXPORT virtual ~Instance();

    virtual bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&) { return false; }
    virtual bool put(JSObject*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&) { return false; }

protected:
    virtual void virtualBegin() { }
    virtual void virtualEnd() { }
    WEBCORE_EXPORT virtual RuntimeObject* newRuntimeObject(JSGlobalObject*);

    RefPtr<RootObject> m_rootObject;

private:
    Weak<RuntimeObject> m_runtimeObject;
};

class Array {
    WTF_MAKE_NONCOPYABLE(Array);
public:
    explicit Array(RefPtr<RootObject>&&);
    virtual ~Array();

    virtual bool setValueAt(JSGlobalObject*, unsigned index, JSValue) const = 0;
    virtual JSValue valueAt(JSGlobalObject*, unsigned index) const = 0;
    virtual unsigned int getLength() const = 0;

protected:
    RefPtr<RootObject> m_rootObject;
};

const char* signatureForParameters(const ArgList&);

} // namespace Bindings

} // namespace JSC
