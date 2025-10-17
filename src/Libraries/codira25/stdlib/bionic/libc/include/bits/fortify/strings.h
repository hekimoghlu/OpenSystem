/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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
#if defined(__BIONIC_FORTIFY)

__BIONIC_FORTIFY_INLINE
void __bionic_bcopy(const void * _Nonnull src, void* _Nonnull const dst __pass_object_size0, size_t len)
        __overloadable
        __clang_error_if(__bos_unevaluated_lt(__bos0(dst), len),
                         "'bcopy' called with size bigger than buffer") {
#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
    size_t bos = __bos0(dst);
    if (!__bos_trivially_ge(bos, len)) {
        __builtin___memmove_chk(dst, src, len, bos);
        return;
    }
#endif
    __builtin_memmove(dst, src, len);
}

__BIONIC_FORTIFY_INLINE
void __bionic_bzero(void* _Nonnull const b __pass_object_size0, size_t len)
        __overloadable
        __clang_error_if(__bos_unevaluated_lt(__bos0(b), len),
                         "'bzero' called with size bigger than buffer") {
#if __BIONIC_FORTIFY_RUNTIME_CHECKS_ENABLED
    size_t bos = __bos0(b);
    if (!__bos_trivially_ge(bos, len)) {
        __builtin___memset_chk(b, 0, len, bos);
        return;
    }
#endif
    __builtin_memset(b, 0, len);
}

#endif /* defined(__BIONIC_FORTIFY) */
