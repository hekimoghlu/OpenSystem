/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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

#include "AbstractModuleRecord.h"
#include "JSDestructibleObject.h"
#include <wtf/FixedVector.h>

namespace JSC {

class JSModuleNamespaceObject final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesGetOwnPropertyNames | OverridesPut | InterceptsGetOwnPropertySlotByIndexEvenWhenLengthIsNotZero | GetOwnPropertySlotIsImpureForPropertyAbsence | IsImmutablePrototypeExoticObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.moduleNamespaceObjectSpace<mode>();
    }

    static JSModuleNamespaceObject* create(JSGlobalObject* globalObject, Structure* structure, AbstractModuleRecord* moduleRecord, Vector<std::pair<Identifier, AbstractModuleRecord::Resolution>>&& resolutions)
    {
        VM& vm = getVM(globalObject);
        JSModuleNamespaceObject* object = new (NotNull, allocateCell<JSModuleNamespaceObject>(vm)) JSModuleNamespaceObject(vm, structure);
        object->finishCreation(globalObject, moduleRecord, WTFMove(resolutions));
        return object;
    }

    JS_EXPORT_PRIVATE static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    JS_EXPORT_PRIVATE static bool getOwnPropertySlotByIndex(JSObject*, JSGlobalObject*, unsigned propertyName, PropertySlot&);
    JS_EXPORT_PRIVATE static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&);
    JS_EXPORT_PRIVATE static bool putByIndex(JSCell*, JSGlobalObject*, unsigned propertyName, JSValue, bool shouldThrow);
    JS_EXPORT_PRIVATE static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    JS_EXPORT_PRIVATE static bool deletePropertyByIndex(JSCell*, JSGlobalObject*, unsigned propertyName);
    JS_EXPORT_PRIVATE static void getOwnPropertyNames(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    JS_EXPORT_PRIVATE static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool shouldThrow);

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    AbstractModuleRecord* moduleRecord() { return m_moduleRecord.get(); }

private:
    JS_EXPORT_PRIVATE JSModuleNamespaceObject(VM&, Structure*);
    JS_EXPORT_PRIVATE void finishCreation(JSGlobalObject*, AbstractModuleRecord*, Vector<std::pair<Identifier, AbstractModuleRecord::Resolution>>&&);
    DECLARE_VISIT_CHILDREN;
    bool getOwnPropertySlotCommon(JSGlobalObject*, PropertyName, PropertySlot&);

    struct ExportEntry {
        Identifier localName;
        WriteBarrier<AbstractModuleRecord> moduleRecord;
    };

    typedef UncheckedKeyHashMap<RefPtr<UniquedStringImpl>, ExportEntry, IdentifierRepHash, HashTraits<RefPtr<UniquedStringImpl>>> ExportMap;

    ExportMap m_exports;
    FixedVector<Identifier> m_names;
    WriteBarrier<AbstractModuleRecord> m_moduleRecord;

    friend size_t cellSize(JSCell*);
};

} // namespace JSC
