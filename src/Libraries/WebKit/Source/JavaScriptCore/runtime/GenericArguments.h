/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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

#include "CagedBarrierPtr.h"
#include "JSObject.h"

namespace JSC {

// This is a mixin for the two kinds of Arguments-class objects that arise when you say
// "arguments" inside a function. This class doesn't show up in the JSCell inheritance hierarchy.
template<typename Type>
class GenericArguments : public JSNonFinalObject {
public:
    typedef JSNonFinalObject Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesPut | InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero;

protected:
    GenericArguments(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    DECLARE_VISIT_CHILDREN;
    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    static bool getOwnPropertySlotByIndex(JSObject*, JSGlobalObject*, unsigned propertyName, PropertySlot&);
    static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    static bool putByIndex(JSCell*, JSGlobalObject*, unsigned propertyName, JSValue, bool shouldThrow);
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    static bool deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned propertyName);
    static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);
    
    void initModifiedArgumentsDescriptor(JSGlobalObject*, unsigned length);
    void initModifiedArgumentsDescriptorIfNecessary(JSGlobalObject*, unsigned length);
    void setModifiedArgumentDescriptor(JSGlobalObject*, unsigned index, unsigned length);
    bool isModifiedArgumentDescriptor(unsigned index, unsigned length);

    void copyToArguments(JSGlobalObject*, JSValue* firstElementDest, unsigned offset, unsigned length);

    using ModifiedArgumentsPtr = CagedBarrierPtr<Gigacage::Primitive, bool>;
    ModifiedArgumentsPtr m_modifiedArgumentsDescriptor;
};

} // namespace JSC
