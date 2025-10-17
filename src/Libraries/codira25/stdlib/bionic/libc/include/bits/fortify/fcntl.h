/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#ifndef _FCNTL_H
#error "Never include this file directly; instead, include <fcntl.h>"
#endif

int __open_2(const char* _Nonnull, int);
int __openat_2(int, const char* _Nonnull, int);
/*
 * These are the easiest way to call the real open even in clang FORTIFY.
 */
int __open_real(const char* _Nonnull, int, ...) __RENAME(open);
int __openat_real(int, const char* _Nonnull, int, ...) __RENAME(openat);

#if defined(__BIONIC_FORTIFY)
#define __open_too_many_args_error "too many arguments"
#define __open_too_few_args_error "called with O_CREAT or O_TMPFILE, but missing mode"
#define __open_useless_modes_warning "has superfluous mode bits; missing O_CREAT or O_TMPFILE?"
/* O_TMPFILE shares bits with O_DIRECTORY. */
#define __open_modes_useful(flags) (((flags) & O_CREAT) || ((flags) & O_TMPFILE) == O_TMPFILE)

__BIONIC_ERROR_FUNCTION_VISIBILITY
int open(const char* _Nonnull pathname, int flags, mode_t modes, ...) __overloadable
        __errorattr(__open_too_many_args_error);

/*
 * pass_object_size serves two purposes here, neither of which involve __bos: it
 * disqualifies this function from having its address taken (so &open works),
 * and it makes overload resolution prefer open(const char *, int) over
 * open(const char *, int, ...).
 */
__BIONIC_FORTIFY_INLINE
int open(const char* _Nonnull const __pass_object_size pathname, int flags)
        __overloadable
        __clang_error_if(__open_modes_useful(flags), "'open' " __open_too_few_args_error) {
#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
    return __open_2(pathname, flags);
#else
    return __open_real(pathname, flags);
#endif
}

__BIONIC_FORTIFY_INLINE
int open(const char* _Nonnull const __pass_object_size pathname, int flags, mode_t modes)
        __overloadable
        __clang_warning_if(!__open_modes_useful(flags) && modes,
                           "'open' " __open_useless_modes_warning) {
    return __open_real(pathname, flags, modes);
}

__BIONIC_ERROR_FUNCTION_VISIBILITY
int openat(int dirfd, const char* _Nonnull pathname, int flags, mode_t modes, ...)
        __overloadable
        __errorattr(__open_too_many_args_error);

__BIONIC_FORTIFY_INLINE
int openat(int dirfd, const char* _Nonnull const __pass_object_size pathname, int flags)
        __overloadable
        __clang_error_if(__open_modes_useful(flags), "'openat' " __open_too_few_args_error) {
#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
    return __openat_2(dirfd, pathname, flags);
#else
    return __openat_real(dirfd, pathname, flags);
#endif
}

__BIONIC_FORTIFY_INLINE
int openat(int dirfd, const char* _Nonnull const __pass_object_size pathname, int flags, mode_t modes)
        __overloadable
        __clang_warning_if(!__open_modes_useful(flags) && modes,
                           "'openat' " __open_useless_modes_warning) {
    return __openat_real(dirfd, pathname, flags, modes);
}

/* Note that open == open64, so we reuse those bits in the open64 variants below.  */

__BIONIC_ERROR_FUNCTION_VISIBILITY
int open64(const char* _Nonnull pathname, int flags, mode_t modes, ...) __overloadable
        __errorattr(__open_too_many_args_error);

__BIONIC_FORTIFY_INLINE
int open64(const char* _Nonnull const __pass_object_size pathname, int flags)
        __overloadable
        __clang_error_if(__open_modes_useful(flags), "'open64' " __open_too_few_args_error) {
    return open(pathname, flags);
}

__BIONIC_FORTIFY_INLINE
int open64(const char* _Nonnull const __pass_object_size pathname, int flags, mode_t modes)
        __overloadable
        __clang_warning_if(!__open_modes_useful(flags) && modes,
                           "'open64' " __open_useless_modes_warning) {
    return open(pathname, flags, modes);
}

__BIONIC_ERROR_FUNCTION_VISIBILITY
int openat64(int dirfd, const char* _Nonnull pathname, int flags, mode_t modes, ...)
        __overloadable
        __errorattr(__open_too_many_args_error);

__BIONIC_FORTIFY_INLINE
int openat64(int dirfd, const char* _Nonnull const __pass_object_size pathname, int flags)
        __overloadable
        __clang_error_if(__open_modes_useful(flags), "'openat64' " __open_too_few_args_error) {
    return openat(dirfd, pathname, flags);
}

__BIONIC_FORTIFY_INLINE
int openat64(int dirfd, const char* _Nonnull const __pass_object_size pathname, int flags, mode_t modes)
        __overloadable
        __clang_warning_if(!__open_modes_useful(flags) && modes,
                           "'openat64' " __open_useless_modes_warning) {
    return openat(dirfd, pathname, flags, modes);
}

#undef __open_too_many_args_error
#undef __open_too_few_args_error
#undef __open_useless_modes_warning
#undef __open_modes_useful
#endif /* defined(__BIONIC_FORTIFY) */
