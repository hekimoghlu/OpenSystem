/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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

#if HAVE(FAST_TLS)

#include <pthread.h>
#include <pthread/tsd_private.h>
#include <wtf/Platform.h>

namespace WTF {

// __PTK_FRAMEWORK_JAVASCRIPTCORE_KEY0-1 is taken by bmalloc, so WTF's KEY0 maps to the
// system's KEY2.
#define WTF_FAST_TLS_KEY0 __PTK_FRAMEWORK_JAVASCRIPTCORE_KEY2
#define WTF_FAST_TLS_KEY1 __PTK_FRAMEWORK_JAVASCRIPTCORE_KEY3

// NOTE: We should manage our use of these keys here. If you want to use a key for something,
// put a #define in here to give your key a symbolic name. This ensures that we don't
// accidentally use the same key for more than one thing.

#define WTF_THREAD_DATA_KEY WTF_FAST_TLS_KEY0
#define WTF_WASM_CONTEXT_KEY WTF_FAST_TLS_KEY1
#define WTF_TESTING_KEY WTF_WASM_CONTEXT_KEY // So far, this key is only used in places that don't do WebAssembly, so it's OK that they share the same key.

#if ENABLE(FAST_TLS_JIT)
inline unsigned fastTLSOffsetForKey(unsigned long slot)
{
    return slot * sizeof(void*);
}
#endif

} // namespace WTF

#if ENABLE(FAST_TLS_JIT)
using WTF::fastTLSOffsetForKey;
#endif

#endif // HAVE(FAST_TLS)

