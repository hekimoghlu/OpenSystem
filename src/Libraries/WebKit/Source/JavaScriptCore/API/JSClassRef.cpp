/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#include "JSClassRef.h"

#include "APICast.h"
#include "InitializeThreading.h"
#include "JSCInlines.h"
#include "JSCallbackObject.h"

using namespace JSC;

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

const JSClassDefinition kJSClassDefinitionEmpty = { 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };

OpaqueJSClass::OpaqueJSClass(const JSClassDefinition* definition, OpaqueJSClass* protoClass) 
    : parentClass(definition->parentClass)
    , prototypeClass(nullptr)
    , initialize(definition->initialize)
    , finalize(definition->finalize)
    , hasProperty(definition->hasProperty)
    , getProperty(definition->getProperty)
    , setProperty(definition->setProperty)
    , deleteProperty(definition->deleteProperty)
    , getPropertyNames(definition->getPropertyNames)
    , callAsFunction(definition->callAsFunction)
    , callAsConstructor(definition->callAsConstructor)
    , hasInstance(definition->hasInstance)
    , convertToType(definition->convertToType)
    , m_className(String::fromUTF8(definition->className))
{
    JSC::initialize();

    if (const JSStaticValue* staticValue = definition->staticValues) {
        m_staticValues = makeUnique<OpaqueJSClassStaticValuesTable>();
        while (staticValue->name) {
            String valueName = String::fromUTF8(staticValue->name);
            if (!valueName.isNull())
                m_staticValues->set(valueName.impl(), makeUnique<StaticValueEntry>(staticValue->getProperty, staticValue->setProperty, staticValue->attributes, valueName));
            ++staticValue;
        }
    }

    if (const JSStaticFunction* staticFunction = definition->staticFunctions) {
        m_staticFunctions = makeUnique<OpaqueJSClassStaticFunctionsTable>();
        while (staticFunction->name) {
            String functionName = String::fromUTF8(staticFunction->name);
            if (!functionName.isNull())
                m_staticFunctions->set(functionName.impl(), makeUnique<StaticFunctionEntry>(staticFunction->callAsFunction, staticFunction->attributes));
            ++staticFunction;
        }
    }
        
    if (protoClass)
        prototypeClass = JSClassRetain(protoClass);
}

OpaqueJSClass::~OpaqueJSClass()
{
    // The empty string is shared across threads & is an identifier, in all other cases we should have done a deep copy in className(), below. 
    ASSERT(!m_className.length() || !m_className.impl()->isAtom());

#ifndef NDEBUG
    if (m_staticValues) {
        OpaqueJSClassStaticValuesTable::const_iterator end = m_staticValues->end();
        for (OpaqueJSClassStaticValuesTable::const_iterator it = m_staticValues->begin(); it != end; ++it)
            ASSERT(!it->key->isAtom());
    }

    if (m_staticFunctions) {
        OpaqueJSClassStaticFunctionsTable::const_iterator end = m_staticFunctions->end();
        for (OpaqueJSClassStaticFunctionsTable::const_iterator it = m_staticFunctions->begin(); it != end; ++it)
            ASSERT(!it->key->isAtom());
    }
#endif
    
    if (prototypeClass)
        JSClassRelease(prototypeClass);
}

Ref<OpaqueJSClass> OpaqueJSClass::createNoAutomaticPrototype(const JSClassDefinition* definition)
{
    return adoptRef(*new OpaqueJSClass(definition, nullptr));
}

Ref<OpaqueJSClass> OpaqueJSClass::create(const JSClassDefinition* clientDefinition)
{
    JSClassDefinition definition = *clientDefinition; // Avoid modifying client copy.

    JSClassDefinition protoDefinition = kJSClassDefinitionEmpty;
    protoDefinition.finalize = nullptr;
    std::swap(definition.staticFunctions, protoDefinition.staticFunctions); // Move static functions to the prototype.
    
    // We are supposed to use JSClassRetain/Release but since we know that we currently have
    // the only reference to this class object we cheat and use a RefPtr instead.
    RefPtr<OpaqueJSClass> protoClass = adoptRef(new OpaqueJSClass(&protoDefinition, nullptr));
    return adoptRef(*new OpaqueJSClass(&definition, protoClass.get()));
}

OpaqueJSClassContextData::OpaqueJSClassContextData(JSC::VM&, OpaqueJSClass* jsClass)
    : m_class(jsClass)
{
    if (jsClass->m_staticValues) {
        staticValues = makeUnique<OpaqueJSClassStaticValuesTable>();
        OpaqueJSClassStaticValuesTable::const_iterator end = jsClass->m_staticValues->end();
        for (OpaqueJSClassStaticValuesTable::const_iterator it = jsClass->m_staticValues->begin(); it != end; ++it) {
            ASSERT(!it->key->isAtom());
            String valueName = it->key->isolatedCopy();
            staticValues->add(valueName.impl(), makeUnique<StaticValueEntry>(it->value->getProperty, it->value->setProperty, it->value->attributes, valueName));
        }
    }

    if (jsClass->m_staticFunctions) {
        staticFunctions = makeUnique<OpaqueJSClassStaticFunctionsTable>();
        OpaqueJSClassStaticFunctionsTable::const_iterator end = jsClass->m_staticFunctions->end();
        for (OpaqueJSClassStaticFunctionsTable::const_iterator it = jsClass->m_staticFunctions->begin(); it != end; ++it) {
            ASSERT(!it->key->isAtom());
            staticFunctions->add(it->key->isolatedCopy(), makeUnique<StaticFunctionEntry>(it->value->callAsFunction, it->value->attributes));
        }
    }
}

OpaqueJSClassContextData& OpaqueJSClass::contextData(JSGlobalObject* globalObject)
{
    auto& contextData = globalObject->contextData(this);
    if (!contextData)
        contextData = makeUnique<OpaqueJSClassContextData>(globalObject->vm(), this);
    return *contextData;
}

String OpaqueJSClass::className()
{
    // Make a deep copy, so that the caller has no chance to put the original into AtomStringTable.
    return m_className.isolatedCopy();
}

OpaqueJSClassStaticValuesTable* OpaqueJSClass::staticValues(JSC::JSGlobalObject* globalObject)
{
    return contextData(globalObject).staticValues.get();
}

OpaqueJSClassStaticFunctionsTable* OpaqueJSClass::staticFunctions(JSC::JSGlobalObject* globalObject)
{
    return contextData(globalObject).staticFunctions.get();
}

JSObject* OpaqueJSClass::prototype(JSGlobalObject* globalObject)
{
    /* Class (C++) and prototype (JS) inheritance are parallel, so:
     *     (C++)      |        (JS)
     *   ParentClass  |   ParentClassPrototype
     *       ^        |          ^
     *       |        |          |
     *  DerivedClass  |  DerivedClassPrototype
     */

    if (!prototypeClass)
        return nullptr;

    OpaqueJSClassContextData& jsClassData = contextData(globalObject);

    if (JSObject* prototype = jsClassData.cachedPrototype.get())
        return prototype;

    // Recursive, but should be good enough for our purposes
    JSObject* prototype = JSCallbackObject<JSNonFinalObject>::create(globalObject, globalObject->callbackObjectStructure(), prototypeClass, &jsClassData); // set jsClassData as the object's private data, so it can clear our reference on destruction
    if (parentClass) {
        if (JSObject* parentPrototype = parentClass->prototype(globalObject))
            prototype->setPrototypeDirect(globalObject->vm(), parentPrototype);
    }

    jsClassData.cachedPrototype = Weak<JSObject>(prototype);
    return prototype;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
