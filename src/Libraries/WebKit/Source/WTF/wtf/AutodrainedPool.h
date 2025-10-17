/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 19, 2022.
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

#if defined(__OBJC__) && !defined(__clang_tapi__)
#error Please use @autoreleasepool instead of AutodrainedPool.
#endif

#include <wtf/Noncopyable.h>

namespace WTF {

// This class allows non-Objective-C C++ code to create an autorelease pool.
// It cannot be used in Objective-C++ code, won't be compiled; instead @autoreleasepool should be used.
// It can be used in cross-platform code; will compile down to nothing for non-Cocoa platforms.

class AutodrainedPool {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(AutodrainedPool);

public:
#if USE(FOUNDATION)
    WTF_EXPORT_PRIVATE AutodrainedPool();
    WTF_EXPORT_PRIVATE ~AutodrainedPool();
#else
    AutodrainedPool() { }
    ~AutodrainedPool() { }
#endif

private:
#if USE(FOUNDATION)
    void* m_autoreleasePool;
#endif
};

} // namespace WTF

using WTF::AutodrainedPool;
