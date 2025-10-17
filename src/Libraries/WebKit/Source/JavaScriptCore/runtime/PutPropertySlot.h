/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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

#include "JSCJSValue.h"
#include "PropertySlot.h"
#include <wtf/Assertions.h>

namespace JSC {
    
class JSObject;
class JSFunction;
    
using CustomAccessorValueFunc = FunctionPtr<CustomAccessorPtrTag, bool(JSGlobalObject*, EncodedJSValue, EncodedJSValue, PropertyName), FunctionAttributes::JITOperation>;

class PutPropertySlot {
public:
    enum Type : uint8_t { Uncachable, ExistingProperty, NewProperty, SetterProperty, CustomValue, CustomAccessor };
    enum Context : uint8_t { UnknownContext, PutById, PutByIdEval };

    PutPropertySlot(JSValue thisValue, bool isStrictMode = false, Context context = UnknownContext, bool isInitialization = false)
        : m_base(nullptr)
        , m_thisValue(thisValue)
        , m_offset(invalidOffset)
        , m_isStrictMode(isStrictMode)
        , m_isInitialization(isInitialization)
        , m_isTaintedByOpaqueObject(false)
        , m_type(Uncachable)
        , m_context(context)
        , m_cacheability(CachingAllowed)
    {
    }

    void setExistingProperty(JSObject* base, PropertyOffset offset)
    {
        m_type = ExistingProperty;
        m_base = base;
        m_offset = offset;
    }

    void setNewProperty(JSObject* base, PropertyOffset offset)
    {
        m_type = NewProperty;
        m_base = base;
        m_offset = offset;
    }

    void setCustomValue(JSObject* base, PutValueFunc function)
    {
        m_type = CustomValue;
        m_base = base;
        m_putFunction = function.get();
    }

    void setCustomAccessor(JSObject* base, PutValueFunc function)
    {
        m_type = CustomAccessor;
        m_base = base;
        m_putFunction = function.get();
    }

    void setCacheableSetter(JSObject* base, PropertyOffset offset)
    {
        m_type = SetterProperty;
        m_base = base;
        m_offset = offset;
    }

    void setThisValue(JSValue thisValue)
    {
        m_thisValue = thisValue;
    }

    void setStrictMode(bool value)
    {
        m_isStrictMode = value;
    }

    CustomAccessorValueFunc customSetter() const
    {
        ASSERT(isCacheableCustom());
        return m_putFunction;
    }

    Type type() const { return m_type; }
    Context context() const { return m_context; }
    JSObject* base() const { return m_base; }
    JSValue thisValue() const { return m_thisValue; }

    bool isStrictMode() const { return m_isStrictMode; }
    bool isCacheablePut() const { return isCacheable() && (m_type == NewProperty || m_type == ExistingProperty); }
    bool isCacheableSetter() const { return isCacheable() && m_type == SetterProperty; }
    bool isCacheableCustom() const { return isCacheable() && (m_type == CustomValue || m_type == CustomAccessor) && !!m_putFunction; }
    bool isCustomAccessor() const { return isCacheable() && m_type == CustomAccessor; }
    bool isInitialization() const { return m_isInitialization; }
    bool isTaintedByOpaqueObject() const { return m_isTaintedByOpaqueObject; }
    void setIsTaintedByOpaqueObject() { m_isTaintedByOpaqueObject = true; }

    PropertyOffset cachedOffset() const
    {
        return m_offset;
    }

    void disableCaching()
    {
        m_cacheability = CachingDisallowed;
    }

private:
    bool isCacheable() const { return m_cacheability == CachingAllowed; }

    JSObject* m_base;
    JSValue m_thisValue;
    PropertyOffset m_offset;
    bool m_isStrictMode : 1;
    bool m_isInitialization : 1;
    bool m_isTaintedByOpaqueObject : 1;
    Type m_type;
    Context m_context;
    CacheabilityType m_cacheability;
    CustomAccessorValueFunc m_putFunction;
};

} // namespace JSC
