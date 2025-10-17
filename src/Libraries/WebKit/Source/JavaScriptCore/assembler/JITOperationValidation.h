/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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

#if ENABLE(JIT_CAGE)
#include <WebKitAdditions/JITCageAdditions.h>
#else // ENABLE(JIT_CAGE)
#if OS(DARWIN)
#define MAP_EXECUTABLE_FOR_JIT MAP_JIT
#define MAP_EXECUTABLE_FOR_JIT_WITH_JIT_CAGE MAP_JIT
#else // OS(DARWIN)
#define MAP_EXECUTABLE_FOR_JIT 0
#define MAP_EXECUTABLE_FOR_JIT_WITH_JIT_CAGE 0
#endif // OS(DARWIN)
#endif

#if !defined(JSC_FORCE_USE_JIT_CAGE)
#define JSC_FORCE_USE_JIT_CAGE 0
#endif

#if !defined(JSC_ALLOW_JIT_CAGE_SPECIFIC_RESERVATION)
#define JSC_ALLOW_JIT_CAGE_SPECIFIC_RESERVATION 1
#endif

namespace JSC {

struct JITOperationAnnotation {
    void* operation;
#if ENABLE(JIT_OPERATION_VALIDATION)
    void* operationWithValidation;
#endif
};

#if ENABLE(JIT_OPERATION_VALIDATION)

#ifndef JSC_DECLARE_JIT_OPERATION_VALIDATION
#define JSC_DECLARE_JIT_OPERATION_VALIDATION(functionName)
#endif

#ifndef JSC_DECLARE_JIT_OPERATION_PROBE_VALIDATION
#define JSC_DECLARE_JIT_OPERATION_PROBE_VALIDATION(functionName)
#endif

#ifndef JSC_DECLARE_JIT_OPERATION_RETURN_VALIDATION
#define JSC_DECLARE_JIT_OPERATION_RETURN_VALIDATION(functionName)
#endif

#ifndef JSC_DEFINE_JIT_OPERATION_VALIDATION
#define JSC_DEFINE_JIT_OPERATION_VALIDATION(functionName) \
    static constexpr auto* functionName##Validate = functionName
#endif

#ifndef JSC_DEFINE_JIT_OPERATION_PROBE_VALIDATION
#define JSC_DEFINE_JIT_OPERATION_PROBE_VALIDATION(functionName) \
    static constexpr auto* functionName##Validate = functionName
#endif

#ifndef JSC_DEFINE_JIT_OPERATION_RETURN_VALIDATION
#define JSC_DEFINE_JIT_OPERATION_RETURN_VALIDATION(functionName) \
    static constexpr auto* functionName##Validate = functionName
#endif

#else // not ENABLE(JIT_OPERATION_VALIDATION)

#define JSC_DECLARE_JIT_OPERATION_VALIDATION(functionName)
#define JSC_DECLARE_JIT_OPERATION_PROBE_VALIDATION(functionName)
#define JSC_DECLARE_JIT_OPERATION_RETURN_VALIDATION(functionName)

#define JSC_DEFINE_JIT_OPERATION_VALIDATION(functionName)
#define JSC_DEFINE_JIT_OPERATION_PROBE_VALIDATION(functionName)
#define JSC_DEFINE_JIT_OPERATION_RETURN_VALIDATION(functionName)

#endif // ENABLE(JIT_OPERATION_VALIDATION)

#define JSC_DECLARE_AND_DEFINE_JIT_OPERATION_VALIDATION(functionName) \
    JSC_DECLARE_JIT_OPERATION_VALIDATION(functionName); \
    JSC_DEFINE_JIT_OPERATION_VALIDATION(functionName)

#define JSC_DECLARE_AND_DEFINE_JIT_OPERATION_PROBE_VALIDATION(functionName) \
    JSC_DECLARE_JIT_OPERATION_PROBE_VALIDATION(functionName); \
    JSC_DEFINE_JIT_OPERATION_PROBE_VALIDATION(functionName)

#define JSC_DECLARE_AND_DEFINE_JIT_OPERATION_RETURN_VALIDATION(functionName) \
    JSC_DECLARE_JIT_OPERATION_RETURN_VALIDATION(functionName); \
    JSC_DEFINE_JIT_OPERATION_RETURN_VALIDATION(functionName)

#ifndef LLINT_DECLARE_ROUTINE_VALIDATE
#define LLINT_DECLARE_ROUTINE_VALIDATE(name) \
    void unused##name##Validate()
#endif

#ifndef LLINT_ROUTINE_VALIDATE
#define LLINT_ROUTINE_VALIDATE(name)  LLInt::getCodeFunctionPtr<CFunctionPtrTag>(name)
#endif

#ifndef LLINT_RETURN_VALIDATE
#define LLINT_RETURN_VALIDATE(name)  LLInt::getCodeFunctionPtr<CFunctionPtrTag>(name)
#endif

#ifndef LLINT_RETURN_WIDE16_VALIDATE
#define LLINT_RETURN_WIDE16_VALIDATE(name)  LLInt::getWide16CodeFunctionPtr<CFunctionPtrTag>(name)
#endif

#ifndef LLINT_RETURN_WIDE32_VALIDATE
#define LLINT_RETURN_WIDE32_VALIDATE(name)  LLInt::getWide32CodeFunctionPtr<CFunctionPtrTag>(name)
#endif

#ifndef JSC_OPERATION_VALIDATION_MACROASSEMBLER_ARM64_SUPPORT
#define JSC_OPERATION_VALIDATION_MACROASSEMBLER_ARM64_SUPPORT() \
    struct NothingToAddForJITOperationValidationMacroAssemblerARM64Support { }
#endif

#ifndef JSC_RETURN_RETAGGED_OPERATION_WITH_VALIDATION
#define JSC_RETURN_RETAGGED_OPERATION_WITH_VALIDATION(operation) \
    return operation.retagged<CFunctionPtrTag>()
#endif

#ifndef JSC_RETURN_RETAGGED_CALL_TARGET_WITH_VALIDATION
#define JSC_RETURN_RETAGGED_CALL_TARGET_WITH_VALIDATION(call) \
    return MacroAssembler::readCallTarget<OperationPtrTag>(call).retagged<CFunctionPtrTag>()
#endif

} // namespace JSC
