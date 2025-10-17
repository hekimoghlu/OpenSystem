/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
#ifndef __BIONIC_PRIVATE_BIONIC_DEFS_H_
#define __BIONIC_PRIVATE_BIONIC_DEFS_H_

/*
 * This label is used to mark libc/libdl symbols that may need to be replaced
 * by native bridge implementation.
 */
#ifdef __ANDROID_NATIVE_BRIDGE__
#define __BIONIC_WEAK_FOR_NATIVE_BRIDGE __attribute__((__weak__, __noinline__))
#define __BIONIC_WEAK_VARIABLE_FOR_NATIVE_BRIDGE __attribute__((__weak__))
#define __BIONIC_WEAK_FOR_NATIVE_BRIDGE_INLINE \
  __BIONIC_WEAK_FOR_NATIVE_BRIDGE extern "C" __LIBC_HIDDEN__
#else
#define __BIONIC_WEAK_FOR_NATIVE_BRIDGE
#define __BIONIC_WEAK_VARIABLE_FOR_NATIVE_BRIDGE
#define __BIONIC_WEAK_FOR_NATIVE_BRIDGE_INLINE static inline
#endif

#endif /* __BIONIC_PRIVATE_BIONIC_DEFS_H_ */
