/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#ifndef _STDIO_H_
#error "Never include this file directly; instead, include <stdio.h>"
#endif

char* _Nullable __fgets_chk(char* _Nonnull, int, FILE* _Nonnull, size_t);

#if __BIONIC_AVAILABILITY_GUARD(24)
size_t __fread_chk(void* _Nonnull, size_t, size_t, FILE* _Nonnull, size_t) __INTRODUCED_IN(24);
size_t __fwrite_chk(const void* _Nonnull, size_t, size_t, FILE* _Nonnull, size_t) __INTRODUCED_IN(24);
#endif /* __BIONIC_AVAILABILITY_GUARD(24) */


#if defined(__BIONIC_FORTIFY) && !defined(__BIONIC_NO_STDIO_FORTIFY)

#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
/* No diag -- clang diagnoses misuses of this on its own.  */
__BIONIC_FORTIFY_INLINE __printflike(3, 0)
int vsnprintf(char* const __BIONIC_COMPLICATED_NULLNESS __pass_object_size dest, size_t size, const char* _Nonnull format, va_list ap)
        __diagnose_as_builtin(__builtin_vsnprintf, 1, 2, 3, 4)
        __overloadable {
    return __builtin___vsnprintf_chk(dest, size, 0, __bos(dest), format, ap);
}

__BIONIC_FORTIFY_INLINE __printflike(2, 0)
int vsprintf(char* const __BIONIC_COMPLICATED_NULLNESS __pass_object_size dest, const char* _Nonnull format, va_list ap) __overloadable {
    return __builtin___vsprintf_chk(dest, 0, __bos(dest), format, ap);
}
#endif

__BIONIC_ERROR_FUNCTION_VISIBILITY
int sprintf(char* __BIONIC_COMPLICATED_NULLNESS dest, const char* _Nonnull format)
    __overloadable
    __enable_if(__bos_unevaluated_lt(__bos(dest), __builtin_strlen(format)),
                "format string will always overflow destination buffer")
    __errorattr("format string will always overflow destination buffer");

#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
__BIONIC_FORTIFY_VARIADIC __printflike(2, 3)
int sprintf(char* const __BIONIC_COMPLICATED_NULLNESS __pass_object_size dest, const char* _Nonnull format, ...) __overloadable {
    va_list va;
    va_start(va, format);
    int result = __builtin___vsprintf_chk(dest, 0, __bos(dest), format, va);
    va_end(va);
    return result;
}

/* No diag -- clang diagnoses misuses of this on its own.  */
__BIONIC_FORTIFY_VARIADIC __printflike(3, 4)
int snprintf(char* const __BIONIC_COMPLICATED_NULLNESS __pass_object_size dest, size_t size, const char* _Nonnull format, ...)
        __diagnose_as_builtin(__builtin_snprintf, 1, 2, 3)
        __overloadable {
    va_list va;
    va_start(va, format);
    int result = __builtin___vsnprintf_chk(dest, size, 0, __bos(dest), format, va);
    va_end(va);
    return result;
}
#endif

#define __bos_trivially_ge_mul(bos_val, size, count) \
  __bos_dynamic_check_impl_and(bos_val, >=, (size) * (count), \
                               !__unsafe_check_mul_overflow(size, count))

__BIONIC_FORTIFY_INLINE
size_t fread(void* const _Nonnull __pass_object_size0 buf, size_t size, size_t count, FILE* _Nonnull stream)
        __overloadable
        __clang_error_if(__unsafe_check_mul_overflow(size, count),
                         "in call to 'fread', size * count overflows")
        __clang_error_if(__bos_unevaluated_lt(__bos0(buf), size * count),
                         "in call to 'fread', size * count is too large for the given buffer") {
#if __ANDROID_API__ >= 24 && __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
    size_t bos = __bos0(buf);

    if (!__bos_trivially_ge_mul(bos, size, count)) {
        return __fread_chk(buf, size, count, stream, bos);
    }
#endif
    return __call_bypassing_fortify(fread)(buf, size, count, stream);
}

__BIONIC_FORTIFY_INLINE
size_t fwrite(const void* const _Nonnull __pass_object_size0 buf, size_t size, size_t count, FILE* _Nonnull stream)
        __overloadable
        __clang_error_if(__unsafe_check_mul_overflow(size, count),
                         "in call to 'fwrite', size * count overflows")
        __clang_error_if(__bos_unevaluated_lt(__bos0(buf), size * count),
                         "in call to 'fwrite', size * count is too large for the given buffer") {
#if __ANDROID_API__ >= 24 && __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
    size_t bos = __bos0(buf);

    if (!__bos_trivially_ge_mul(bos, size, count)) {
        return __fwrite_chk(buf, size, count, stream, bos);
    }
#endif
    return __call_bypassing_fortify(fwrite)(buf, size, count, stream);
}
#undef __bos_trivially_ge_mul

__BIONIC_FORTIFY_INLINE
char* _Nullable fgets(char* const _Nonnull __pass_object_size dest, int size, FILE* _Nonnull stream)
        __overloadable
        __clang_error_if(size < 0, "in call to 'fgets', size should not be negative")
        __clang_error_if(__bos_unevaluated_lt(__bos(dest), size),
                         "in call to 'fgets', size is larger than the destination buffer") {
#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
    size_t bos = __bos(dest);

    if (!__bos_dynamic_check_impl_and(bos, >=, (size_t)size, size >= 0)) {
        return __fgets_chk(dest, size, stream, bos);
    }
#endif
    return __call_bypassing_fortify(fgets)(dest, size, stream);
}

#endif /* defined(__BIONIC_FORTIFY) */
