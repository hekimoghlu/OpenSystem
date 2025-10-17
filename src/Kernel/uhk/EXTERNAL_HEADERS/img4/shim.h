/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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
#ifndef __IMG4_SHIM_H
#define __IMG4_SHIM_H

#ifndef __IMG4_INDIRECT
#error "Please #include <img4/firmware.h> instead of this file directly"
#endif // __IMG4_INDIRECT

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#if KERNEL
#define IMG4_TARGET_SHIM_FOUND 1
#define IMG4_TARGET_XNU 1

#if __has_include(<img4/shim_xnu.h>)
#include <img4/shim_xnu.h>
#endif

#if XNU_KERNEL_PRIVATE
#define IMG4_TARGET_XNU_PROPER 1
#else
#define IMG4_TARGET_XNU_PROPER 0
#endif
#endif // KERNEL


#if !IMG4_TARGET_SHIM_FOUND
#if EFI
#define IMG4_TARGET_EFI 1
#if __has_include(<img4/shim_efi.h>)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpragma-pack"
#include <img4/shim_efi.h>
#pragma clang diagnostic pop
#endif // __has_include(<img4/shim_efi.h>)
#elif SEP // EFI
#define IMG4_TARGET_SEP 1
#include <img4/shim_sep.h>
#else // EFI
#define IMG4_TARGET_DARWIN 1
#if __has_include(<img4/shim_darwin.h>)
#include <img4/shim_darwin.h>
#endif
#endif // EFI
#endif // !IMG4_TARGET_SHIM_FOUND

#if IMG4_TARGET_XNU || IMG4_TARGET_DARWIN
#define IMG4_TARGET_DARWIN_GENERIC 1
#endif

#endif // __IMG4_SHIM_H
