/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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

#include "CacheableIdentifier.h"
#include "CallVariant.h"

namespace JSC {

class CallLinkInfo;
class DirectCallLinkInfo;
class OptimizingCallLinkInfo;
class StructureStubInfo;

enum class GetByKind {
    ById,
    ByVal,
    TryById,
    ByIdWithThis,
    ByIdDirect,
    ByValWithThis,
    PrivateName,
    PrivateNameById,
};

enum class PutByKind {
    ByIdStrict,
    ByIdSloppy,
    ByValStrict,
    ByValSloppy,
    ByIdDirectStrict,
    ByIdDirectSloppy,
    ByValDirectStrict,
    ByValDirectSloppy,
    DefinePrivateNameById,
    DefinePrivateNameByVal,
    SetPrivateNameById,
    SetPrivateNameByVal,
};

enum class DelByKind {
    ByIdStrict,
    ByIdSloppy,
    ByValStrict,
    ByValSloppy,
};

enum class InByKind {
    ById,
    ByVal,
    PrivateName
};

void repatchArrayGetByVal(JSGlobalObject*, CodeBlock*, JSValue base, JSValue index, StructureStubInfo&, GetByKind);
void repatchGetBy(JSGlobalObject*, CodeBlock*, JSValue, CacheableIdentifier, const PropertySlot&, StructureStubInfo&, GetByKind);
void repatchArrayPutByVal(JSGlobalObject*, CodeBlock*, JSValue base, JSValue index, StructureStubInfo&, PutByKind);
void repatchPutBy(JSGlobalObject*, CodeBlock*, JSValue, Structure*, CacheableIdentifier, const PutPropertySlot&, StructureStubInfo&, PutByKind);
void repatchDeleteBy(JSGlobalObject*, CodeBlock*, DeletePropertySlot&, JSValue, Structure*, CacheableIdentifier, StructureStubInfo&, DelByKind, ECMAMode);
void repatchArrayInByVal(JSGlobalObject*, CodeBlock*, JSValue base, JSValue index, StructureStubInfo&, InByKind);
void repatchInBy(JSGlobalObject*, CodeBlock*, JSObject*, CacheableIdentifier, bool wasFound, const PropertySlot&, StructureStubInfo&, InByKind);
void repatchHasPrivateBrand(JSGlobalObject*, CodeBlock*, JSObject*, CacheableIdentifier, bool wasFound,  StructureStubInfo&);
void repatchCheckPrivateBrand(JSGlobalObject*, CodeBlock*, JSObject*, CacheableIdentifier, StructureStubInfo&);
void repatchSetPrivateBrand(JSGlobalObject*, CodeBlock*, JSObject*, Structure*, CacheableIdentifier, StructureStubInfo&);
void repatchInstanceOf(JSGlobalObject*, CodeBlock*, JSValue, JSValue prototype, StructureStubInfo&, bool wasFound);
void linkMonomorphicCall(VM&, JSCell*, CallLinkInfo&, CodeBlock*, JSObject* callee, CodePtr<JSEntryPtrTag>);
void linkDirectCall(DirectCallLinkInfo&, CodeBlock*, CodePtr<JSEntryPtrTag>);
void linkPolymorphicCall(VM&, JSCell*, CallFrame*, CallLinkInfo&, CallVariant);
void resetGetBy(CodeBlock*, StructureStubInfo&, GetByKind);
void resetPutBy(CodeBlock*, StructureStubInfo&, PutByKind);
void resetDelBy(CodeBlock*, StructureStubInfo&, DelByKind);
void resetInBy(CodeBlock*, StructureStubInfo&, InByKind);
void resetHasPrivateBrand(CodeBlock*, StructureStubInfo&);
void resetInstanceOf(CodeBlock*, StructureStubInfo&);
void resetCheckPrivateBrand(CodeBlock*, StructureStubInfo&);
void resetSetPrivateBrand(CodeBlock*, StructureStubInfo&);

void repatchGetBySlowPathCall(CodeBlock*, StructureStubInfo&, GetByKind);
void repatchPutBySlowPathCall(CodeBlock*, StructureStubInfo&, PutByKind);
void repatchInBySlowPathCall(CodeBlock*, StructureStubInfo&, InByKind);

void ftlThunkAwareRepatchCall(CodeBlock*, CodeLocationCall<JSInternalPtrTag>, CodePtr<CFunctionPtrTag> newCalleeFunction);
CodePtr<JSEntryPtrTag> jsToWasmICCodePtr(CodeSpecializationKind, JSObject* callee);

} // namespace JSC
