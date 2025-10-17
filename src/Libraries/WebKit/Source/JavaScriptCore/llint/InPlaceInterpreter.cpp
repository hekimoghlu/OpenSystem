/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include "InPlaceInterpreter.h"

#if ENABLE(WEBASSEMBLY)

#include "ArithProfile.h"
#include "CodeBlock.h"
#include "JSCConfig.h"
#include "LLIntPCRanges.h"
#include "LLIntSlowPaths.h"
#include "LLIntThunks.h"
#include "Opcode.h"
#include "WriteBarrier.h"

namespace JSC { namespace IPInt {

#define VALIDATE_IPINT_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_unreachable_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 256, #name); \
} while (false);

#define VALIDATE_IPINT_0xFB_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_struct_new_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 256, #name); \
} while (false);

#define VALIDATE_IPINT_0xFC_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_i32_trunc_sat_f32_s_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 256, #name); \
} while (false);

#define VALIDATE_IPINT_SIMD_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_simd_v128_load_mem_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 256, #name); \
} while (false);

#define VALIDATE_IPINT_ATOMIC_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_memory_atomic_notify_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 256, #name); \
} while (false);

#define VALIDATE_IPINT_ARGUMINT_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_argumINT_a0_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 64, #name); \
} while (false);

#define VALIDATE_IPINT_SLOW_PATH(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_local_get_slow_path_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 256, #name); \
} while (false);

#define VALIDATE_IPINT_MINT_CALL_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_mint_a0_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 64, #name); \
} while (false);

#define VALIDATE_IPINT_MINT_RETURN_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_mint_r0_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 64, #name); \
} while (false);

#define VALIDATE_IPINT_UINT_OPCODE(opcode, name) \
do { \
    void* base = reinterpret_cast<void*>(ipint_uint_r0_validate); \
    void* ptr = reinterpret_cast<void*>(ipint_ ## name ## _validate); \
    void* untaggedBase = CodePtr<CFunctionPtrTag>::fromTaggedPtr(base).template untaggedPtr<>(); \
    void* untaggedPtr = CodePtr<CFunctionPtrTag>::fromTaggedPtr(ptr).template untaggedPtr<>(); \
    RELEASE_ASSERT_WITH_MESSAGE((char*)(untaggedPtr) - (char*)(untaggedBase) == opcode * 64, #name); \
} while (false);

void initialize()
{
#if !ENABLE(C_LOOP) && ((CPU(ADDRESS64) && (CPU(ARM64) || CPU(X86_64))) || (CPU(ADDRESS32) && CPU(ARM_THUMB2)))
    FOR_EACH_IPINT_OPCODE(VALIDATE_IPINT_OPCODE);
    FOR_EACH_IPINT_0xFB_OPCODE(VALIDATE_IPINT_0xFB_OPCODE);
    FOR_EACH_IPINT_0xFC_TRUNC_OPCODE(VALIDATE_IPINT_0xFC_OPCODE);
    FOR_EACH_IPINT_SIMD_OPCODE(VALIDATE_IPINT_SIMD_OPCODE);
    FOR_EACH_IPINT_ATOMIC_OPCODE(VALIDATE_IPINT_ATOMIC_OPCODE);

    FOR_EACH_IPINT_ARGUMINT_OPCODE(VALIDATE_IPINT_ARGUMINT_OPCODE);
    FOR_EACH_IPINT_SLOW_PATH(VALIDATE_IPINT_SLOW_PATH);
    FOR_EACH_IPINT_MINT_CALL_OPCODE(VALIDATE_IPINT_MINT_CALL_OPCODE);
    FOR_EACH_IPINT_MINT_RETURN_OPCODE(VALIDATE_IPINT_MINT_RETURN_OPCODE);
    FOR_EACH_IPINT_UINT_OPCODE(VALIDATE_IPINT_UINT_OPCODE);
#else
    RELEASE_ASSERT_NOT_REACHED("IPInt only supports ARM64 and X86_64 (for now).");
#endif
}

} }

#endif // ENABLE(WEBASSEMBLY)
