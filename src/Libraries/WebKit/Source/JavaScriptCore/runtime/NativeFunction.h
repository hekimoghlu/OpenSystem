/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

#include "JSCJSValue.h"
#include "JSCPtrTag.h"
#include <wtf/FunctionPtr.h>
#include <wtf/Hasher.h>

namespace JSC {

class CallFrame;

using NativeFunction = FunctionPtr<CFunctionPtrTag, EncodedJSValue(JSGlobalObject*, CallFrame*), FunctionAttributes::JSCHostCall>;

struct NativeFunctionHash {
    static unsigned hash(NativeFunction key) { return computeHash(key); }
    static bool equal(NativeFunction a, NativeFunction b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

using TaggedNativeFunction = FunctionPtr<HostFunctionPtrTag, EncodedJSValue(JSGlobalObject*, CallFrame*), FunctionAttributes::JSCHostCall>;

struct TaggedNativeFunctionHash {
    static unsigned hash(TaggedNativeFunction key) { return computeHash(key); }
    static bool equal(TaggedNativeFunction a, TaggedNativeFunction b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

static_assert(sizeof(NativeFunction) == sizeof(void*));
static_assert(sizeof(TaggedNativeFunction) == sizeof(void*));

static inline TaggedNativeFunction toTagged(NativeFunction function)
{
    return function.retagged<HostFunctionPtrTag>();
}

} // namespace JSC

namespace WTF {

inline void add(Hasher& hasher, JSC::NativeFunction key)
{
    add(hasher, key.taggedPtr());
}

template<typename> struct DefaultHash;
template<> struct DefaultHash<JSC::NativeFunction> : JSC::NativeFunctionHash { };

inline void add(Hasher& hasher, JSC::TaggedNativeFunction key)
{
    add(hasher, key.taggedPtr());
}

template<typename> struct DefaultHash;
template<> struct DefaultHash<JSC::TaggedNativeFunction> : JSC::TaggedNativeFunctionHash { };

} // namespace WTF
