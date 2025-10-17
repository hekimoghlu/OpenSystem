/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#include "DocumentInlines.h"
#include "ElementInlines.h"
#include "JSNodeCustomInlines.h"
#include "WebCoreOpaqueRoot.h"

namespace WebCore {

template<typename Visitor>
ALWAYS_INLINE void addWebCoreOpaqueRoot(Visitor& visitor, WebCoreOpaqueRoot root)
{
    visitor.addOpaqueRoot(root.pointer());
}

template<typename Visitor, typename ImplType>
ALWAYS_INLINE void addWebCoreOpaqueRoot(Visitor& visitor, ImplType* impl)
{
    addWebCoreOpaqueRoot(visitor, root(impl));
}

template<typename Visitor, typename ImplType>
ALWAYS_INLINE void addWebCoreOpaqueRoot(Visitor& visitor, ImplType& impl)
{
    addWebCoreOpaqueRoot(visitor, root(&impl));
}

template<typename Visitor>
ALWAYS_INLINE bool containsWebCoreOpaqueRoot(Visitor& visitor, WebCoreOpaqueRoot root)
{
    return visitor.containsOpaqueRoot(root.pointer());
}

template<typename Visitor, typename ImplType>
ALWAYS_INLINE bool containsWebCoreOpaqueRoot(Visitor& visitor, ImplType& impl)
{
    return containsWebCoreOpaqueRoot(visitor, root(&impl));
}

template<typename Visitor, typename ImplType>
ALWAYS_INLINE bool containsWebCoreOpaqueRoot(Visitor& visitor, ImplType* impl)
{
    return containsWebCoreOpaqueRoot(visitor, root(impl));
}

} // namespace WebCore
