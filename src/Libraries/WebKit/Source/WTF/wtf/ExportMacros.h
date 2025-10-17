/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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

#include <wtf/Platform.h>

// Different platforms have different defaults for symbol visibility. Usually
// the compiler and the linker just take care of it. However for references to
// runtime routines from JIT stubs, it matters to be able to declare a symbol as
// being local to the target being generated, and thus not subject to (e.g.) ELF
// symbol interposition rules.

#if USE(DECLSPEC_ATTRIBUTE)
#define HAVE_INTERNAL_VISIBILITY 1
#define WTF_INTERNAL
#elif USE(VISIBILITY_ATTRIBUTE)
#define HAVE_INTERNAL_VISIBILITY 1
#define WTF_INTERNAL __attribute__((visibility("hidden")))
#else
#define WTF_INTERNAL
#endif

#if USE(DECLSPEC_ATTRIBUTE)
#define WTF_EXPORT_DECLARATION __declspec(dllexport)
#define WTF_IMPORT_DECLARATION __declspec(dllimport)
#elif USE(VISIBILITY_ATTRIBUTE)
#define WTF_EXPORT_DECLARATION __attribute__((visibility("default")))
#define WTF_IMPORT_DECLARATION WTF_EXPORT_DECLARATION
#else
#define WTF_EXPORT_DECLARATION
#define WTF_IMPORT_DECLARATION
#endif

#if !defined(WTF_EXPORT_PRIVATE)

#if defined(BUILDING_WTF) || defined(STATICALLY_LINKED_WITH_WTF)
#define WTF_EXPORT_PRIVATE WTF_EXPORT_DECLARATION
#else
#define WTF_EXPORT_PRIVATE WTF_IMPORT_DECLARATION
#endif

#endif
