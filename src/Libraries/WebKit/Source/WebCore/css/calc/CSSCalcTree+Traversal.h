/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

#include "CSSCalcTree.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {
namespace CSSCalc {

// MARK: Traversal

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
        void operator()(const AtomString& root)
        {
            functor(root);
        }
        void operator()(const MQ::MediaProgressProviding* root)
        {
            functor(root);
        }
        void operator()(const CQ::ContainerProgressProviding* root)
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

// MARK: - forAllChildNodes

// `forAllChildNodes` will call the provided `functor` on all direct `Child` typed children of the provided node. If a child is of type `ChildOrNone` or `std::optional<Child>`, the functor will be called on the unwrapped `Child` if and only if that is what the type is holding.

template<typename F, Leaf Op> void forAllChildNodes(const Op&, const F&)
{
    // No children.
}

template<typename F, typename Op> void forAllChildNodes(const Op& root, const F& functor)
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
            if (root)
                functor(*root);
        }
        void operator()(const ChildOrNone& root)
        {
            WTF::switchOn(root,
                [&](const Child& root) { functor(root); },
                [&](const CSS::Keyword::None&) { }
            );
        }
        void operator()(const Child& root)
        {
            functor(root);
        }
        void operator()(const AtomString&)
        {
        }
        void operator()(const MQ::MediaProgressProviding*)
        {
        }
        void operator()(const CQ::ContainerProgressProviding*)
        {
        }
        void operator()(const Random::CachingOptions&)
        {
        }
    };
    auto caller = Caller { functor };
    WTF::apply([&](const auto& ...x) { (..., caller(x)); }, root);
}

template<typename F> void forAllChildNodes(const Child& root, const F& functor)
{
    WTF::switchOn(root, [&](const auto& root) { forAllChildNodes(*root, functor); });
}

} // namespace CSSCalc
} // namespace WebCore
