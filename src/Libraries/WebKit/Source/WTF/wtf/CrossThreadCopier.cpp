/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#include <wtf/CrossThreadCopier.h>

namespace WTF {

// Test CrossThreadCopier using static_assert.

// Verify that ThreadSafeRefCounted objects get handled correctly.
class CopierThreadSafeRefCountedTest : public ThreadSafeRefCounted<CopierThreadSafeRefCountedTest> {
};

static_assert((std::is_same<RefPtr<CopierThreadSafeRefCountedTest>, CrossThreadCopier<RefPtr<CopierThreadSafeRefCountedTest>>::Type>::value), "RefPtrTest");
static_assert((std::is_same<RefPtr<CopierThreadSafeRefCountedTest>, CrossThreadCopier<CopierThreadSafeRefCountedTest*>::Type>::value), "RawPointerTest");
static_assert((std::is_same<Ref<CopierThreadSafeRefCountedTest>, CrossThreadCopier<Ref<CopierThreadSafeRefCountedTest>>::Type>::value), "RawPointerTest");

template<typename T> struct CrossThreadCopierBase<false, false, T*> {
    typedef int Type;
};

// Verify that RefCounted objects only match the above templates which expose Type as int.
class CopierRefCountedTest : public RefCounted<CopierRefCountedTest> {
};

static_assert((std::is_same<int, CrossThreadCopier<CopierRefCountedTest*>::Type>::value), "CrossThreadCopier specialization improperly applied to raw pointer of a RefCounted (but not ThreadSafeRefCounted) type");

} // namespace WTF

