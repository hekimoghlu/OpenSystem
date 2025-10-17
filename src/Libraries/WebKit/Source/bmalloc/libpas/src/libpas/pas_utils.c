/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_utils.h"

#include "pas_lock.h"
#include "pas_log.h"
#include "pas_string_stream.h"
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

#if PAS_X86_64

#define PAS_FATAL_CRASH_INST "int3"

#define CRASH_GPR0 "r11"
#define CRASH_GPR1 "r10"
#define CRASH_GPR2 "r9"
#define CRASH_GPR3 "r8"
#define CRASH_GPR4 "r15"
#define CRASH_GPR5 "r14"
#define CRASH_GPR6 "r13"

#elif PAS_ARM64

#if !defined(PAS_FATAL_CRASH_CODE)
#if BASAN_ENABLED
#define PAS_FATAL_CRASH_INST "brk #0x0"
#else
#define PAS_FATAL_CRASH_INST "brk #0xc471"
#endif
#endif

#define CRASH_GPR0 "x16"
#define CRASH_GPR1 "x17"
#define CRASH_GPR2 "x19" // We skip x18, which is reserved on ARM64 for platform use.
#define CRASH_GPR3 "x20"
#define CRASH_GPR4 "x21"
#define CRASH_GPR5 "x22"
#define CRASH_GPR6 "x23"

#endif

#if defined(PAS_BMALLOC) && PAS_BMALLOC
#if defined(__has_include)
#if __has_include(<WebKitAdditions/pas_utils_additions.c>) && !PAS_ENABLE_TESTING
#include <WebKitAdditions/pas_utils_additions.c>
#endif // __has_include(<WebKitAdditions/pas_utils_additions.c>) && !PAS_ENABLE_TESTING
#endif // defined(__has_include)
#endif // defined(PAS_BMALLOC) && PAS_BMALLOC

#if PAS_X86_64 || PAS_ARM64

#if PAS_OS(DARWIN) && PAS_VA_OPT_SUPPORTED

PAS_NEVER_INLINE PAS_NO_RETURN void pas_crash_with_info_impl1(uint64_t reason, uint64_t misc1)
{
    register uint64_t reasonGPR asm(CRASH_GPR0) = reason;
    register uint64_t misc1GPR asm(CRASH_GPR1) = misc1;
    __asm__ volatile (PAS_FATAL_CRASH_INST : : "r"(reasonGPR), "r"(misc1GPR));
    __builtin_unreachable();
}

PAS_NEVER_INLINE PAS_NO_RETURN void pas_crash_with_info_impl2(uint64_t reason, uint64_t misc1, uint64_t misc2)
{
    register uint64_t reasonGPR asm(CRASH_GPR0) = reason;
    register uint64_t misc1GPR asm(CRASH_GPR1) = misc1;
    register uint64_t misc2GPR asm(CRASH_GPR2) = misc2;
    __asm__ volatile (PAS_FATAL_CRASH_INST : : "r"(reasonGPR), "r"(misc1GPR), "r"(misc2GPR));
    __builtin_unreachable();
}

PAS_NEVER_INLINE PAS_NO_RETURN void pas_crash_with_info_impl3(uint64_t reason, uint64_t misc1, uint64_t misc2, uint64_t misc3)
{
    register uint64_t reasonGPR asm(CRASH_GPR0) = reason;
    register uint64_t misc1GPR asm(CRASH_GPR1) = misc1;
    register uint64_t misc2GPR asm(CRASH_GPR2) = misc2;
    register uint64_t misc3GPR asm(CRASH_GPR3) = misc3;
    __asm__ volatile (PAS_FATAL_CRASH_INST : : "r"(reasonGPR), "r"(misc1GPR), "r"(misc2GPR), "r"(misc3GPR));
    __builtin_unreachable();
}

PAS_NEVER_INLINE PAS_NO_RETURN void pas_crash_with_info_impl4(uint64_t reason, uint64_t misc1, uint64_t misc2, uint64_t misc3, uint64_t misc4)
{
    register uint64_t reasonGPR asm(CRASH_GPR0) = reason;
    register uint64_t misc1GPR asm(CRASH_GPR1) = misc1;
    register uint64_t misc2GPR asm(CRASH_GPR2) = misc2;
    register uint64_t misc3GPR asm(CRASH_GPR3) = misc3;
    register uint64_t misc4GPR asm(CRASH_GPR4) = misc4;
    __asm__ volatile (PAS_FATAL_CRASH_INST : : "r"(reasonGPR), "r"(misc1GPR), "r"(misc2GPR), "r"(misc3GPR), "r"(misc4GPR));
    __builtin_unreachable();
}

PAS_NEVER_INLINE PAS_NO_RETURN void pas_crash_with_info_impl5(uint64_t reason, uint64_t misc1, uint64_t misc2, uint64_t misc3, uint64_t misc4, uint64_t misc5)
{
    register uint64_t reasonGPR asm(CRASH_GPR0) = reason;
    register uint64_t misc1GPR asm(CRASH_GPR1) = misc1;
    register uint64_t misc2GPR asm(CRASH_GPR2) = misc2;
    register uint64_t misc3GPR asm(CRASH_GPR3) = misc3;
    register uint64_t misc4GPR asm(CRASH_GPR4) = misc4;
    register uint64_t misc5GPR asm(CRASH_GPR5) = misc5;
    __asm__ volatile (PAS_FATAL_CRASH_INST : : "r"(reasonGPR), "r"(misc1GPR), "r"(misc2GPR), "r"(misc3GPR), "r"(misc4GPR), "r"(misc5GPR));
    __builtin_unreachable();
}

PAS_NEVER_INLINE PAS_NO_RETURN void pas_crash_with_info_impl6(uint64_t reason, uint64_t misc1, uint64_t misc2, uint64_t misc3, uint64_t misc4, uint64_t misc5, uint64_t misc6)
{
    register uint64_t reasonGPR asm(CRASH_GPR0) = reason;
    register uint64_t misc1GPR asm(CRASH_GPR1) = misc1;
    register uint64_t misc2GPR asm(CRASH_GPR2) = misc2;
    register uint64_t misc3GPR asm(CRASH_GPR3) = misc3;
    register uint64_t misc4GPR asm(CRASH_GPR4) = misc4;
    register uint64_t misc5GPR asm(CRASH_GPR5) = misc5;
    register uint64_t misc6GPR asm(CRASH_GPR6) = misc6;
    __asm__ volatile (PAS_FATAL_CRASH_INST : : "r"(reasonGPR), "r"(misc1GPR), "r"(misc2GPR), "r"(misc3GPR), "r"(misc4GPR), "r"(misc5GPR), "r"(misc6GPR));
    __builtin_unreachable();
}

#endif /* PAS_OS(DARWIN) && PAS_VA_OPT_SUPPORTED */

PAS_NEVER_INLINE PAS_NO_RETURN static void pas_crash_with_info_impl(uint64_t reason, uint64_t misc1, uint64_t misc2, uint64_t misc3, uint64_t misc4, uint64_t misc5, uint64_t misc6)
{
    register uint64_t reasonGPR asm(CRASH_GPR0) = reason;
    register uint64_t misc1GPR asm(CRASH_GPR1) = misc1;
    register uint64_t misc2GPR asm(CRASH_GPR2) = misc2;
    register uint64_t misc3GPR asm(CRASH_GPR3) = misc3;
    register uint64_t misc4GPR asm(CRASH_GPR4) = misc4;
    register uint64_t misc5GPR asm(CRASH_GPR5) = misc5;
    register uint64_t misc6GPR asm(CRASH_GPR6) = misc6;
    __asm__ volatile (PAS_FATAL_CRASH_INST : : "r"(reasonGPR), "r"(misc1GPR), "r"(misc2GPR), "r"(misc3GPR), "r"(misc4GPR), "r"(misc5GPR), "r"(misc6GPR));
    __builtin_trap();
}

#else

PAS_IGNORE_WARNINGS_BEGIN("unused-parameter")
PAS_NEVER_INLINE PAS_NO_RETURN static void pas_crash_with_info_impl(uint64_t reason, uint64_t misc1, uint64_t misc2, uint64_t misc3, uint64_t misc4, uint64_t misc5, uint64_t misc6) { __builtin_trap(); }
PAS_IGNORE_WARNINGS_END

#endif

void pas_panic(const char* format, ...)
{
    static const bool fast_panic = false;
    if (!fast_panic) {
        va_list arg_list;
        pas_log("[%d] pas panic: ", getpid());
        va_start(arg_list, format);
        pas_vlog(format, arg_list);
        pas_crash_with_info_impl((uint64_t)format, 0, 0, 0, 0, 0, 0);
    }
    __builtin_trap();
}

#if PAS_ENABLE_TESTING
PAS_NEVER_INLINE void pas_report_assertion_failed(
    const char* filename, int line, const char* function, const char* expression)
{
    pas_log("[%d] pas panic: ", getpid());
    pas_log("%s:%d: %s: assertion %s failed.\n", filename, line, function, expression);
}
#endif /* PAS_ENABLE_TESTING */

void pas_assertion_failed_no_inline(const char* filename, int line, const char* function, const char* expression)
{
    pas_log("[%d] pas assertion failed: ", getpid());
    pas_log("%s:%d: %s: assertion %s failed.\n", filename, line, function, expression);
    pas_crash_with_info_impl((uint64_t)filename, (uint64_t)line, (uint64_t)function, (uint64_t)expression, 0xbeefbff0, 42, 1337);
}

void pas_assertion_failed_no_inline_with_extra_detail(const char* filename, int line, const char* function, const char* expression, uint64_t extra)
{
    pas_log("[%d] pas assertion failed (with extra detail): ", getpid());
    pas_log("%s:%d: %s: assertion %s failed. Extra data: %" PRIu64 ".\n", filename, line, function, expression, extra);
    pas_crash_with_info_impl((uint64_t)filename, (uint64_t)line, (uint64_t)function, (uint64_t)expression, extra, 1337, 0xbeef0bff);
}

void pas_panic_on_out_of_memory_error(void)
{
    __builtin_trap();
}

static void (*deallocation_did_fail_callback)(const char* reason, void* begin);

PAS_NO_RETURN PAS_NEVER_INLINE void pas_deallocation_did_fail(const char *reason, uintptr_t begin)
{
    if (deallocation_did_fail_callback)
        deallocation_did_fail_callback(reason, (void*)begin);
    pas_panic("deallocation did fail at %p: %s\n", (void*)begin, reason);
}

void pas_set_deallocation_did_fail_callback(void (*callback)(const char* reason, void* begin))
{
    deallocation_did_fail_callback = callback;
}

static void (*reallocation_did_fail_callback)(const char* reason,
                                              void* source_heap,
                                              void* target_heap,
                                              void* old_ptr,
                                              size_t old_size,
                                              size_t new_size);

PAS_NO_RETURN PAS_NEVER_INLINE void pas_reallocation_did_fail(const char *reason,
                                                              void* source_heap,
                                                              void* target_heap,
                                                              void* old_ptr,
                                                              size_t old_size,
                                                              size_t new_size)
{
    if (reallocation_did_fail_callback) {
        reallocation_did_fail_callback(
            reason, source_heap, target_heap, old_ptr, old_size, new_size);
    }
    pas_panic("reallocation did fail with source_heap = %p, target_heap = %p, "
              "old_ptr = %p, old_size = %zu, new_size = %zu: %s\n",
              source_heap, target_heap, old_ptr, old_size, new_size,
              reason);
}

void pas_set_reallocation_did_fail_callback(void (*callback)(const char* reason,
                                                             void* source_heap,
                                                             void* target_heap,
                                                             void* old_ptr,
                                                             size_t old_size,
                                                             size_t new_count))
{
    reallocation_did_fail_callback = callback;
}

#endif /* LIBPAS_ENABLED */
