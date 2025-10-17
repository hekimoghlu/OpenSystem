/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include "WasmGlobal.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCJSValueInlines.h"
#include "JSWebAssemblyGlobal.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyRuntimeError.h"
#include "WasmTypeDefinitionInlines.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace Wasm {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Global);

JSValue Global::get(JSGlobalObject* globalObject) const
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    switch (m_type.kind) {
    case TypeKind::I32:
        return jsNumber(std::bit_cast<int32_t>(static_cast<uint32_t>(m_value.m_primitive)));
    case TypeKind::I64:
        RELEASE_AND_RETURN(throwScope, JSBigInt::makeHeapBigIntOrBigInt32(globalObject, static_cast<int64_t>(m_value.m_primitive)));
    case TypeKind::F32:
        return jsNumber(purifyNaN(static_cast<double>(std::bit_cast<float>(static_cast<uint32_t>(m_value.m_primitive)))));
    case TypeKind::F64:
        return jsNumber(purifyNaN(std::bit_cast<double>(m_value.m_primitive)));
    case TypeKind::V128:
        throwException(globalObject, throwScope, createJSWebAssemblyRuntimeError(globalObject, vm, "Cannot get value of v128 global"_s));
        return { };
    case TypeKind::Exn:
    case TypeKind::Externref:
    case TypeKind::Funcref:
    case TypeKind::Ref:
    case TypeKind::RefNull: {
        if (UNLIKELY(isExnref(m_type))) {
            throwException(globalObject, throwScope, createJSWebAssemblyRuntimeError(globalObject, vm, "Cannot get value of exnref global"_s));
            return { };
        }
        return m_value.m_externref.get();
    }
    default:
        return jsUndefined();
    }
}

void Global::set(JSGlobalObject* globalObject, JSValue argument)
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    ASSERT(m_mutability != Wasm::Immutable);
    switch (m_type.kind) {
    case TypeKind::I32: {
        int32_t value = argument.toInt32(globalObject);
        RETURN_IF_EXCEPTION(throwScope, void());
        m_value.m_primitive = static_cast<uint64_t>(static_cast<uint32_t>(value));
        break;
    }
    case TypeKind::I64: {
        int64_t value = argument.toBigInt64(globalObject);
        RETURN_IF_EXCEPTION(throwScope, void());
        m_value.m_primitive = static_cast<uint64_t>(value);
        break;
    }
    case TypeKind::F32: {
        float value = argument.toFloat(globalObject);
        RETURN_IF_EXCEPTION(throwScope, void());
        m_value.m_primitive = static_cast<uint64_t>(std::bit_cast<uint32_t>(value));
        break;
    }
    case TypeKind::F64: {
        double value = argument.toNumber(globalObject);
        RETURN_IF_EXCEPTION(throwScope, void());
        m_value.m_primitive = std::bit_cast<uint64_t>(value);
        break;
    }
    case TypeKind::V128: {
        throwTypeError(globalObject, throwScope, "Cannot set value of v128 global"_s);
        return;
    }
    case Wasm::TypeKind::Ref:
    case Wasm::TypeKind::RefNull:
    case Wasm::TypeKind::Externref:
    case Wasm::TypeKind::Funcref: {
        if (isExternref(m_type)) {
            RELEASE_ASSERT(m_owner);
            if (!m_type.isNullable() && argument.isNull()) {
                throwTypeError(globalObject, throwScope, "Non-null Externref cannot be null"_s);
                return;
            }
            m_value.m_externref.set(m_owner->vm(), m_owner, argument);
        } else if (isFuncref(m_type) || (isRefWithTypeIndex(m_type) && TypeInformation::get(m_type.index).is<FunctionSignature>())) {
            RELEASE_ASSERT(m_owner);
            WebAssemblyFunction* wasmFunction = nullptr;
            WebAssemblyWrapperFunction* wasmWrapperFunction = nullptr;
            if (!isWebAssemblyHostFunction(argument, wasmFunction, wasmWrapperFunction) && (!m_type.isNullable() || !argument.isNull())) {
                throwTypeError(globalObject, throwScope, "Argument value did not match the reference type"_s);
                return;
            }

            if (isRefWithTypeIndex(m_type) && !argument.isNull()) {
                Wasm::TypeIndex paramIndex = m_type.index;
                Wasm::TypeIndex argIndex = wasmFunction ? wasmFunction->typeIndex() : wasmWrapperFunction->typeIndex();
                if (paramIndex != argIndex) {
                    throwTypeError(globalObject, throwScope, "Argument value did not match the reference type"_s);
                    return;
                }
            }
            m_value.m_externref.set(m_owner->vm(), m_owner, argument);
        } else if (isExnref(m_type)) {
            throwTypeError(globalObject, throwScope, "Cannot set value of exnref global"_s);
            return;
        } else {
            JSValue internref = Wasm::internalizeExternref(argument);
            if (!Wasm::TypeInformation::castReference(internref, m_type.isNullable(), m_type.index)) {
                // FIXME: provide a better error message here
                // https://bugs.webkit.org/show_bug.cgi?id=247746
                throwTypeError(globalObject, throwScope, "Argument value did not match the reference type"_s);
                return;
            }
            m_value.m_externref.set(m_owner->vm(), m_owner, internref);
        }
        break;
    }
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

template<typename Visitor>
void Global::visitAggregateImpl(Visitor& visitor)
{
    if (isRefType(m_type)) {
        RELEASE_ASSERT(m_owner);
        visitor.append(m_value.m_externref);
    }
}

DEFINE_VISIT_AGGREGATE(Global);

} } // namespace JSC::Global

#endif // ENABLE(WEBASSEMBLY)
