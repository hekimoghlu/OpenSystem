/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#import "objc_class.h"
#import "objc_utility.h"

namespace JSC {

namespace Bindings {

class ObjcClass;

class ObjcInstance : public Instance {
public:
    static Ref<ObjcInstance> create(ObjectStructPtr, RefPtr<RootObject>&&);
    virtual ~ObjcInstance();
    
    static void setGlobalException(NSString*, JSGlobalObject* exceptionEnvironment = 0); // A null exceptionEnvironment means the exception should propogate to any execution environment.

    virtual Class* getClass() const;
        
    virtual JSValue valueOf(JSGlobalObject*) const;
    virtual JSValue defaultValue(JSGlobalObject*, PreferredPrimitiveType) const;

    virtual JSValue getMethod(JSGlobalObject*, PropertyName);
    JSValue invokeObjcMethod(JSGlobalObject*, CallFrame*, ObjcMethod* method);
    virtual JSValue invokeMethod(JSGlobalObject*, CallFrame*, RuntimeMethod* method);
    virtual bool supportsInvokeDefaultMethod() const;
    virtual JSValue invokeDefaultMethod(JSGlobalObject*, CallFrame*);

    JSValue getValueOfUndefinedField(JSGlobalObject*, PropertyName) const;
    virtual bool setValueOfUndefinedField(JSGlobalObject*, PropertyName, JSValue);

    ObjectStructPtr getObject() const { return _instance.get(); }
    
    JSValue stringValue(JSGlobalObject*) const;
    JSValue numberValue(JSGlobalObject*) const;
    JSValue booleanValue() const;

    static bool isInStringValue();

protected:
    virtual void virtualBegin();
    virtual void virtualEnd();

private:
    friend class ObjcField;
    static void moveGlobalExceptionToExecState(JSGlobalObject*);

    ObjcInstance(ObjectStructPtr, RefPtr<RootObject>&&);

    virtual RuntimeObject* newRuntimeObject(JSGlobalObject*);

    RetainPtr<ObjectStructPtr> _instance;
    mutable ObjcClass* _class { nullptr };
    void* m_autoreleasePool { nullptr };
    int _beginCount { 0 };
};

} // namespace Bindings

} // namespace JSC
