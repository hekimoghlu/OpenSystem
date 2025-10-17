/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

#include <wtf/Atomics.h>

namespace WTF {

struct NoTailCalls {
    ~NoTailCalls()
    {
        compilerFence();
    }
};

} // namespace WTF

// Use this macro when you don't want to perform any tail calls from within a given scope.
// For example, if you don't want the function foo to do a tail call to bar or baz, you'd do:
// int foo(bool b)
// {
//     NO_TAIL_CALLS();
//     if (b)
//         return baz();
//     return bar();
// }
//
// This is helpful when bar or baz have other callers that are allowed to tail call it.

#define NO_TAIL_CALLS() WTF::NoTailCalls _noTailCalls_
