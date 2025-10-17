/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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

#include <wtf/Compiler.h>

// See <https://clang.llvm.org/docs/ThreadSafetyAnalysis.html> for details.

#if COMPILER(CLANG)
#define WTF_THREAD_ANNOTATION_ATTRIBUTE(x)  __attribute__((x))
#else
#define WTF_THREAD_ANNOTATION_ATTRIBUTE(x)
#endif

#define WTF_ACQUIRES_CAPABILITY_IF(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_capability(__VA_ARGS__))
#define WTF_ACQUIRES_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(__VA_ARGS__))
#define WTF_ACQUIRES_SHARED_CAPABILITY_IF(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_shared_capability(__VA_ARGS__))
#define WTF_ACQUIRES_SHARED_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(acquire_shared_capability(__VA_ARGS__))
#define WTF_ASSERTS_ACQUIRED_CAPABILITY(x) WTF_THREAD_ANNOTATION_ATTRIBUTE(assert_capability(x))
#define WTF_ASSERTS_ACQUIRED_SHARED_CAPABILITY(x) WTF_THREAD_ANNOTATION_ATTRIBUTE(assert_shared_capability(x))
#define WTF_CAPABILITY(name) WTF_THREAD_ANNOTATION_ATTRIBUTE(capability(name))
#define WTF_CAPABILITY_SCOPED_LOCK WTF_THREAD_ANNOTATION_ATTRIBUTE(scoped_lockable)
#define WTF_EXCLUDES_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(locks_excluded(__VA_ARGS__))
#define WTF_GUARDED_BY_CAPABILITY(x) WTF_THREAD_ANNOTATION_ATTRIBUTE(guarded_by(x))
#define WTF_IGNORES_THREAD_SAFETY_ANALYSIS WTF_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis)
#define WTF_POINTEE_GUARDED_BY_CAPABILITY(x) WTF_THREAD_ANNOTATION_ATTRIBUTE(pt_guarded_by(x))
#define WTF_RELEASES_GENERIC_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(release_generic_capability(__VA_ARGS__))
#define WTF_RELEASES_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(release_capability(__VA_ARGS__))
#define WTF_RELEASES_SHARED_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(release_shared_capability(__VA_ARGS__))
#define WTF_REQUIRES_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(__VA_ARGS__))
#define WTF_REQUIRES_SHARED_CAPABILITY(...) WTF_THREAD_ANNOTATION_ATTRIBUTE(requires_shared_capability(__VA_ARGS__))
#define WTF_RETURNS_CAPABILITY(x) WTF_THREAD_ANNOTATION_ATTRIBUTE(lock_returned(x))

// Using WTF_CAPABILITY_LOCK is a common use-case. Introduce terms containing "LOCK" for maximum readability.
#define WTF_ACQUIRES_LOCK_IF(...) WTF_ACQUIRES_CAPABILITY_IF(__VA_ARGS__)
#define WTF_ACQUIRES_LOCK(...)  WTF_ACQUIRES_CAPABILITY(__VA_ARGS__)
#define WTF_ACQUIRES_SHARED_LOCK_IF(...) WTF_ACQUIRES_SHARED_CAPABILITY_IF(__VA_ARGS__)
#define WTF_ACQUIRES_SHARED_LOCK(...) WTF_ACQUIRES_SHARED_CAPABILITY(__VA_ARGS__)
#define WTF_ASSERTS_ACQUIRED_LOCK(x) WTF_ASSERTS_ACQUIRED_CAPABILITY(x)
#define WTF_ASSERTS_ACQUIRED_SHARED_LOCK(x) WTF_ASSERTS_ACQUIRED_SHARED_CAPABILITY(x)
#define WTF_CAPABILITY_LOCK WTF_CAPABILITY("lock")
#define WTF_EXCLUDES_LOCK(...) WTF_EXCLUDES_CAPABILITY(__VA_ARGS__)
#define WTF_GUARDED_BY_LOCK(x) WTF_GUARDED_BY_CAPABILITY(x)
#define WTF_POINTEE_GUARDED_BY_LOCK(x) WTF_POINTEE_GUARDED_BY_CAPABILITY(x)
#define WTF_RELEASES_GENERIC_LOCK(...) WTF_RELEASES_GENERIC_CAPABILITY(__VA_ARGS__)
#define WTF_RELEASES_LOCK(...) WTF_RELEASES_CAPABILITY(__VA_ARGS__)
#define WTF_RELEASES_SHARED_LOCK(...) WTF_RELEASES_SHARED_CAPABILITY(__VA_ARGS__)
#define WTF_REQUIRES_LOCK(...) WTF_REQUIRES_CAPABILITY(__VA_ARGS__)
#define WTF_REQUIRES_SHARED_LOCK(...) WTF_REQUIRES_SHARED_CAPABILITY(__VA_ARGS__)
#define WTF_RETURNS_LOCK(x) WTF_RETURNS_CAPABILITY(x)
