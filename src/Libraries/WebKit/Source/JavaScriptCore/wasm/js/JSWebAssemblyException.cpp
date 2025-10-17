/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#include "JSWebAssemblyException.h"
#include "WasmExceptionType.h"
#include "WasmTypeDefinition.h"

#if ENABLE(WEBASSEMBLY)

#include "AuxiliaryBarrierInlines.h"
#include "JSBigInt.h"
#include "JSCJSValueInlines.h"
#include "JSCellInlines.h"
#include "JSWebAssemblyHelpers.h"
#include "StructureInlines.h"

namespace JSC {

const ClassInfo JSWebAssemblyException::s_info = { "WebAssembly.Exception"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWebAssemblyException) };

Structure* JSWebAssemblyException::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ErrorInstanceType, StructureFlags), info());
}

JSWebAssemblyException::JSWebAssemblyException(VM& vm, Structure* structure, Ref<const Wasm::Tag>&& tag, FixedVector<uint64_t>&& payload)
    : Base(vm, structure)
    , m_tag(WTFMove(tag))
    , m_payload(WTFMove(payload))
{
}

void JSWebAssemblyException::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    vm.heap.reportExtraMemoryAllocated(this, payload().byteSize());
}

template<typename Visitor>
void JSWebAssemblyException::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    Base::visitChildren(cell, visitor);

    auto* exception = jsCast<JSWebAssemblyException*>(cell);
    const auto& tagType = exception->tag().type();
    unsigned offset = 0;
    for (unsigned i = 0; i < tagType.argumentCount(); ++i) {
        if (isRefType(tagType.argumentType(i)))
            visitor.append(std::bit_cast<WriteBarrier<Unknown>>(exception->payload()[offset]));
        offset += tagType.argumentType(i).kind == Wasm::TypeKind::V128 ? 2 : 1;
    }
    visitor.reportExtraMemoryVisited(exception->payload().size());
}

DEFINE_VISIT_CHILDREN(JSWebAssemblyException);

void JSWebAssemblyException::destroy(JSCell* cell)
{
    static_cast<JSWebAssemblyException*>(cell)->JSWebAssemblyException::~JSWebAssemblyException();
}

JSValue JSWebAssemblyException::getArg(JSGlobalObject* globalObject, unsigned i) const
{
    const auto& tagType = tag().type();
    ASSERT(i < tagType.argumentCount());

    // It feels like maybe we should throw an exception here, but as far as I can tell,
    // the current draft spec just asserts that we can't getArg a v128. Maybe we can
    // revisit this later.
    RELEASE_ASSERT(tagType.argumentType(i).kind != Wasm::TypeKind::V128);

    unsigned offset = 0;
    for (unsigned j = 0; j < i; ++j)
        offset += tagType.argumentType(j).kind == Wasm::TypeKind::V128 ? 2 : 1;
    return toJSValue(globalObject, tagType.argumentType(i), payload()[offset]);
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
