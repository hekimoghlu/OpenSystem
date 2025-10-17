/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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

#include "ConstructData.h"
#include "JSCast.h"
#include <wtf/CompactPtr.h>
#include <wtf/PtrTag.h>

#if HAVE(36BIT_ADDRESS)
#define CLASS_INFO_ALIGNMENT alignas(16)
#else
#define CLASS_INFO_ALIGNMENT
#endif

namespace WTF {
class PrintStream;
};

namespace JSC {

class HeapAnalyzer;
class JSArrayBufferView;
class Snippet;
struct HashTable;

#define METHOD_TABLE_ENTRY(method) \
    WTF_VTBL_FUNCPTR_PTRAUTH_STR("MethodTable." #method) method

struct MethodTable {
    using DestroyFunctionPtr = void (*)(JSCell*);
    DestroyFunctionPtr METHOD_TABLE_ENTRY(destroy);

    using GetCallDataFunctionPtr = CallData (*)(JSCell*);
    GetCallDataFunctionPtr METHOD_TABLE_ENTRY(getCallData);

    using GetConstructDataFunctionPtr = CallData (*)(JSCell*);
    GetConstructDataFunctionPtr METHOD_TABLE_ENTRY(getConstructData);

    using PutFunctionPtr = bool (*)(JSCell*, JSGlobalObject*, PropertyName propertyName, JSValue, PutPropertySlot&);
    PutFunctionPtr METHOD_TABLE_ENTRY(put);

    using PutByIndexFunctionPtr = bool (*)(JSCell*, JSGlobalObject*, unsigned propertyName, JSValue, bool shouldThrow);
    PutByIndexFunctionPtr METHOD_TABLE_ENTRY(putByIndex);

    using DeletePropertyFunctionPtr = bool (*)(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&);
    DeletePropertyFunctionPtr METHOD_TABLE_ENTRY(deleteProperty);

    using DeletePropertyByIndexFunctionPtr = bool (*)(JSCell*, JSGlobalObject*, unsigned);
    DeletePropertyByIndexFunctionPtr METHOD_TABLE_ENTRY(deletePropertyByIndex);

    using GetOwnPropertySlotFunctionPtr = bool (*)(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&);
    GetOwnPropertySlotFunctionPtr METHOD_TABLE_ENTRY(getOwnPropertySlot);

    using GetOwnPropertySlotByIndexFunctionPtr = bool (*)(JSObject*, JSGlobalObject*, unsigned, PropertySlot&);
    GetOwnPropertySlotByIndexFunctionPtr METHOD_TABLE_ENTRY(getOwnPropertySlotByIndex);

    using GetOwnPropertyNamesFunctionPtr = void (*)(JSObject*, JSGlobalObject*, PropertyNameArray&, DontEnumPropertiesMode);
    GetOwnPropertyNamesFunctionPtr METHOD_TABLE_ENTRY(getOwnPropertyNames);
    GetOwnPropertyNamesFunctionPtr METHOD_TABLE_ENTRY(getOwnSpecialPropertyNames);

    using CustomHasInstanceFunctionPtr = bool (*)(JSObject*, JSGlobalObject*, JSValue);
    CustomHasInstanceFunctionPtr METHOD_TABLE_ENTRY(customHasInstance);

    using DefineOwnPropertyFunctionPtr = bool (*)(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool);
    DefineOwnPropertyFunctionPtr METHOD_TABLE_ENTRY(defineOwnProperty);

    using PreventExtensionsFunctionPtr = bool (*)(JSObject*, JSGlobalObject*);
    PreventExtensionsFunctionPtr METHOD_TABLE_ENTRY(preventExtensions);

    using IsExtensibleFunctionPtr = bool (*)(JSObject*, JSGlobalObject*);
    IsExtensibleFunctionPtr METHOD_TABLE_ENTRY(isExtensible);

    using SetPrototypeFunctionPtr = bool (*)(JSObject*, JSGlobalObject*, JSValue, bool shouldThrowIfCantSet);
    SetPrototypeFunctionPtr METHOD_TABLE_ENTRY(setPrototype);

    using GetPrototypeFunctionPtr = JSValue (*)(JSObject*, JSGlobalObject*);
    GetPrototypeFunctionPtr METHOD_TABLE_ENTRY(getPrototype);

    using DumpToStreamFunctionPtr = void (*)(const JSCell*, PrintStream&);
    DumpToStreamFunctionPtr METHOD_TABLE_ENTRY(dumpToStream);

    using AnalyzeHeapFunctionPtr = void (*)(JSCell*, HeapAnalyzer&);
    AnalyzeHeapFunctionPtr METHOD_TABLE_ENTRY(analyzeHeap);

    using EstimatedSizeFunctionPtr = size_t (*)(JSCell*, VM&);
    EstimatedSizeFunctionPtr METHOD_TABLE_ENTRY(estimatedSize);

    using VisitChildrenFunctionPtr = void (*)(JSCell*, SlotVisitor&);
    VisitChildrenFunctionPtr METHOD_TABLE_ENTRY(visitChildrenWithSlotVisitor);

    using VisitChildrenWithAbstractSlotVisitorPtr = void (*)(JSCell*, AbstractSlotVisitor&);
    VisitChildrenWithAbstractSlotVisitorPtr METHOD_TABLE_ENTRY(visitChildrenWithAbstractSlotVisitor);

    ALWAYS_INLINE void visitChildren(JSCell* cell, SlotVisitor& visitor) const { visitChildrenWithSlotVisitor(cell, visitor); }
    ALWAYS_INLINE void visitChildren(JSCell* cell, AbstractSlotVisitor& visitor) const { visitChildrenWithAbstractSlotVisitor(cell, visitor); }

    using VisitOutputConstraintsPtr = void (*)(JSCell*, SlotVisitor&);
    VisitOutputConstraintsPtr METHOD_TABLE_ENTRY(visitOutputConstraintsWithSlotVisitor);

    using VisitOutputConstraintsWithAbstractSlotVisitorPtr = void (*)(JSCell*, AbstractSlotVisitor&);
    VisitOutputConstraintsWithAbstractSlotVisitorPtr METHOD_TABLE_ENTRY(visitOutputConstraintsWithAbstractSlotVisitor);

    ALWAYS_INLINE void visitOutputConstraints(JSCell* cell, SlotVisitor& visitor) const { visitOutputConstraintsWithSlotVisitor(cell, visitor); }
    ALWAYS_INLINE void visitOutputConstraints(JSCell* cell, AbstractSlotVisitor& visitor) const { visitOutputConstraintsWithAbstractSlotVisitor(cell, visitor); }
};

#undef METHOD_TABLE_ENTRY

#define CREATE_METHOD_TABLE(ClassName) \
    JSCastingHelpers::InheritsTraits<ClassName>::typeRange, \
    { \
        &ClassName::destroy, \
        &ClassName::getCallData, \
        &ClassName::getConstructData, \
        &ClassName::put, \
        &ClassName::putByIndex, \
        &ClassName::deleteProperty, \
        &ClassName::deletePropertyByIndex, \
        &ClassName::getOwnPropertySlot, \
        &ClassName::getOwnPropertySlotByIndex, \
        &ClassName::getOwnPropertyNames, \
        &ClassName::getOwnSpecialPropertyNames, \
        &ClassName::customHasInstance, \
        &ClassName::defineOwnProperty, \
        &ClassName::preventExtensions, \
        &ClassName::isExtensible, \
        &ClassName::setPrototype, \
        &ClassName::getPrototype, \
        &ClassName::dumpToStream, \
        &ClassName::analyzeHeap, \
        &ClassName::estimatedSize, \
        &ClassName::visitChildren, \
        &ClassName::visitChildren, \
        &ClassName::visitOutputConstraints, \
        &ClassName::visitOutputConstraints, \
    }, \
    sizeof(ClassName), \
    ClassName::isResizableOrGrowableSharedTypedArray, \

struct CLASS_INFO_ALIGNMENT ClassInfo {
    WTF_ALLOW_STRUCT_COMPACT_POINTERS;

    using CheckJSCastSnippetFunctionPtr = Ref<Snippet> (*)(void);

    // A string denoting the class name. Example: "Window".
    ASCIILiteral className;
    // Pointer to the class information of the base class.
    // nullptrif there is none.
    const ClassInfo* parentClass;
    const HashTable* staticPropHashTable;
    CheckJSCastSnippetFunctionPtr checkSubClassSnippet;
    const std::optional<JSTypeRange> inheritsJSTypeRange; // This is range of JSTypes for doing inheritance checking. Has the form: [firstJSType, lastJSType] (inclusive).
    MethodTable methodTable;
    const unsigned staticClassSize;
    const bool isResizableOrGrowableSharedTypedArray;

    static constexpr ptrdiff_t offsetOfParentClass()
    {
        return OBJECT_OFFSETOF(ClassInfo, parentClass);
    }

    bool isSubClassOf(const ClassInfo* other) const
    {
        for (const ClassInfo* ci = this; ci; ci = ci->parentClass) {
            if (ci == other)
                return true;
        }
        return false;
    }

    JS_EXPORT_PRIVATE void dump(PrintStream&) const;

    JS_EXPORT_PRIVATE bool hasStaticPropertyWithAnyOfAttributes(uint8_t attributes) const;
};

} // namespace JSC
