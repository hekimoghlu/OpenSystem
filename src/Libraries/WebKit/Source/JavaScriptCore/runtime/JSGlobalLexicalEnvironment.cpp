/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include "JSGlobalLexicalEnvironment.h"

#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSGlobalLexicalEnvironment::s_info = { "JSGlobalLexicalEnvironment"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSGlobalLexicalEnvironment) };

void JSGlobalLexicalEnvironment::destroy(JSCell* cell)
{
    static_cast<JSGlobalLexicalEnvironment*>(cell)->JSGlobalLexicalEnvironment::~JSGlobalLexicalEnvironment();
}

bool JSGlobalLexicalEnvironment::getOwnPropertySlot(JSObject* object, JSGlobalObject*, PropertyName propertyName, PropertySlot& slot)
{
    JSGlobalLexicalEnvironment* thisObject = jsCast<JSGlobalLexicalEnvironment*>(object);
    return symbolTableGet(thisObject, propertyName, slot);
}

bool JSGlobalLexicalEnvironment::put(JSCell* cell, JSGlobalObject* globalObject, PropertyName propertyName, JSValue value, PutPropertySlot& slot)
{
    JSGlobalLexicalEnvironment* thisObject = jsCast<JSGlobalLexicalEnvironment*>(cell);
    ASSERT(!Heap::heap(value) || Heap::heap(value) == Heap::heap(thisObject));
    bool alwaysThrowWhenAssigningToConstProperty = true;
    bool ignoreConstAssignmentError = slot.isInitialization();
    bool putResult = false;
    symbolTablePutTouchWatchpointSet(thisObject, globalObject, propertyName, value, alwaysThrowWhenAssigningToConstProperty, ignoreConstAssignmentError, putResult);
    return putResult;
}

bool JSGlobalLexicalEnvironment::isConstVariable(UniquedStringImpl* impl)
{
    SymbolTableEntry entry = symbolTable()->get(impl);
    ASSERT(!entry.isNull());
    return entry.isReadOnly();
}

} // namespace JSC
