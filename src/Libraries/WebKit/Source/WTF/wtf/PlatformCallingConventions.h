/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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

#ifndef WTF_PLATFORM_GUARD_AGAINST_INDIRECT_INCLUSION
#error "Please #include <wtf/Platform.h> instead of this file directly."
#endif

/* Macros for specifing specific calling conventions. */

#if OS(WINDOWS)
#define SYSV_ABI __attribute__((sysv_abi))
#else
#define SYSV_ABI
#endif

#if CPU(X86)
#define JSC_HOST_CALL_ATTRIBUTES __attribute__ ((fastcall))
#else
#define JSC_HOST_CALL_ATTRIBUTES SYSV_ABI
#endif

#define JSC_ANNOTATE_HOST_FUNCTION(functionId, function)

#define JSC_DEFINE_HOST_FUNCTION_WITH_ATTRIBUTES(functionName, attributes, parameters) \
    JSC_ANNOTATE_HOST_FUNCTION(_JITTarget_##functionName, static_cast<JSC::EncodedJSValue(*)parameters>(functionName)); \
    attributes JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES functionName parameters
#define JSC_DEFINE_HOST_FUNCTION(functionName, parameters) \
    JSC_DEFINE_HOST_FUNCTION_WITH_ATTRIBUTES(functionName, , parameters)
#define JSC_DECLARE_HOST_FUNCTION(functionName) \
    JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES functionName(JSC::JSGlobalObject*, JSC::CallFrame*)

#if CPU(X86) && OS(WINDOWS)
#define CALLING_CONVENTION_IS_STDCALL 1
#else
#define CALLING_CONVENTION_IS_STDCALL 0
#endif

#ifndef CDECL
#if !OS(WINDOWS)
#define CDECL
#else
#define CDECL __attribute__ ((cdecl))
#endif
#endif

#if CPU(X86)
#define WTF_COMPILER_SUPPORTS_FASTCALL_CALLING_CONVENTION 1
#ifndef FASTCALL
#define FASTCALL  __attribute__ ((fastcall))
#endif
#else
#define WTF_COMPILER_SUPPORTS_FASTCALL_CALLING_CONVENTION 0
#endif

#if ENABLE(JIT) && CALLING_CONVENTION_IS_STDCALL
#define JIT_OPERATION_ATTRIBUTES CDECL
#elif OS(WINDOWS)
#define JIT_OPERATION_ATTRIBUTES SYSV_ABI
#else
#define JIT_OPERATION_ATTRIBUTES
#endif

#if ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

#if ENABLE(JIT_OPERATION_VALIDATION)
#define JSC_ANNOTATE_JIT_OPERATION_EXTRAS(validateFunction) (void*)validateFunction
#else
#define JSC_ANNOTATE_JIT_OPERATION_EXTRAS(validateFunction)
#endif

#define JSC_ANNOTATE_JIT_OPERATION_INTERNAL(function) \
    constexpr JSC::JITOperationAnnotation _JITTargetID_##function __attribute__((used, section("__DATA_CONST,__jsc_ops"))) = { (void*)function, JSC_ANNOTATE_JIT_OPERATION_EXTRAS(function##Validate) };

#define JSC_ANNOTATE_JIT_OPERATION(function) \
    JSC_DECLARE_AND_DEFINE_JIT_OPERATION_VALIDATION(function); \
    JSC_ANNOTATE_JIT_OPERATION_INTERNAL(function)

#define JSC_ANNOTATE_JIT_OPERATION_PROBE(function) \
    JSC_DECLARE_AND_DEFINE_JIT_OPERATION_PROBE_VALIDATION(function); \
    JSC_ANNOTATE_JIT_OPERATION_INTERNAL(function)

#define JSC_ANNOTATE_JIT_OPERATION_RETURN(function) \
    JSC_DECLARE_AND_DEFINE_JIT_OPERATION_RETURN_VALIDATION(function); \
    JSC_ANNOTATE_JIT_OPERATION_INTERNAL(function)

#else // not (ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY))

#define JSC_ANNOTATE_JIT_OPERATION(function)
#define JSC_ANNOTATE_JIT_OPERATION_PROBE(function)
#define JSC_ANNOTATE_JIT_OPERATION_RETURN(function)

#endif // ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY)

#define JSC_DEFINE_JIT_OPERATION_WITHOUT_VARIABLE_IMPL(functionName, returnType, parameters) \
    returnType JIT_OPERATION_ATTRIBUTES functionName parameters

#define JSC_DEFINE_JIT_OPERATION_WITH_ATTRIBUTES_IMPL(functionName, attributes, returnType, parameters) \
    JSC_ANNOTATE_JIT_OPERATION(functionName); \
    attributes returnType JIT_OPERATION_ATTRIBUTES functionName parameters

#define JSC_DEFINE_JIT_OPERATION_IMPL(functionName, returnType, parameters) \
    JSC_DEFINE_JIT_OPERATION_WITH_ATTRIBUTES_IMPL(functionName, , returnType, parameters)

#define JSC_DECLARE_JIT_OPERATION_WITH_ATTRIBUTES_IMPL(functionName, attributes, returnType, parameters) \
    extern "C" attributes returnType JIT_OPERATION_ATTRIBUTES functionName parameters REFERENCED_FROM_ASM WTF_INTERNAL; \
    JSC_DECLARE_JIT_OPERATION_VALIDATION(functionName) \

#define JSC_DECLARE_JIT_OPERATION_IMPL(functionName, returnType, parameters) \
    JSC_DECLARE_JIT_OPERATION_WITH_ATTRIBUTES_IMPL(functionName, , returnType, parameters)

#define JSC_DECLARE_JIT_OPERATION_WITHOUT_WTF_INTERNAL_IMPL(functionName, returnType, parameters) \
    returnType JIT_OPERATION_ATTRIBUTES functionName parameters REFERENCED_FROM_ASM

#define JSC_DECLARE_CUSTOM_GETTER(functionName) JSC_DECLARE_JIT_OPERATION_WITHOUT_WTF_INTERNAL_IMPL(functionName, JSC::EncodedJSValue, (JSC::JSGlobalObject*, JSC::EncodedJSValue, JSC::PropertyName))
#define JSC_DECLARE_CUSTOM_SETTER(functionName) JSC_DECLARE_JIT_OPERATION_WITHOUT_WTF_INTERNAL_IMPL(functionName, bool, (JSC::JSGlobalObject*, JSC::EncodedJSValue, JSC::EncodedJSValue, JSC::PropertyName))
#define JSC_DEFINE_CUSTOM_GETTER(functionName, parameters) JSC_DEFINE_JIT_OPERATION_WITHOUT_VARIABLE_IMPL(functionName, JSC::EncodedJSValue, parameters)
#define JSC_DEFINE_CUSTOM_SETTER(functionName, parameters) JSC_DEFINE_JIT_OPERATION_WITHOUT_VARIABLE_IMPL(functionName, bool, parameters)
