/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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

#include "CalculationTree.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {
namespace Calculation {

// MARK: - forAllChildren

// `forAllChildren` will call the provided `functor` on all direct children of the provided node. This will include values of type `Child`, `ChildOrNone` and `std::optional<Child>`.

template<typename F, Leaf Op> void forAllChildren(const auto&, const F&)
{
    // No children.
}

template<typename F, typename Op> void forAllChildren(const Op& root, const F& functor)
{
    struct Caller {
        const F& functor;

        void operator()(const Children& children)
        {
            for (auto& child : children)
                functor(child);
        }
        void operator()(const std::optional<Child>& root)
        {
            functor(root);
        }
        void operator()(const ChildOrNone& root)
        {
            functor(root);
        }
        void operator()(const Child& root)
        {
            functor(root);
        }
        void operator()(const double& root)
        {
            functor(root);
        }
        void operator()(const Random::CachingOptions& root)
        {
            functor(root);
        }
    };
    auto caller = Caller { functor };
    WTF::apply([&](const auto& ...x) { (..., caller(x)); }, root);
}

template<typename F> void forAllChildren(const Child& root, const F& functor)
{
    WTF::switchOn(root, [&](const auto& root) { forAllChildren(*root, functor); });
}

} // namespace Calculation
} // namespace WebCore
