/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

#include "JSDOMBinding.h"
#include "JSNode.h"
#include "WebCoreOpaqueRoot.h"

namespace JSC {
namespace JSCastingHelpers {

template<>
struct InheritsTraits<WebCore::JSNode> {
    static constexpr std::optional<JSTypeRange> typeRange { { static_cast<JSType>(WebCore::JSNodeType), static_cast<JSType>(WebCore::JSNodeType + WebCore::JSNodeTypeMask) } };
    static_assert(std::numeric_limits<uint8_t>::max() == typeRange->last);
    template<typename From>
    static inline bool inherits(From* from)
    {
        return inheritsJSTypeImpl<WebCore::JSNode>(from, *typeRange);
    }
};

} // namespace JSCastingHelpers
} // namespace JSC

namespace WebCore {

WEBCORE_EXPORT JSC::JSValue createWrapper(JSC::JSGlobalObject*, JSDOMGlobalObject*, Ref<Node>&&);
WEBCORE_EXPORT JSC::JSObject* getOutOfLineCachedWrapper(JSDOMGlobalObject*, Node&);

inline JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Node& node)
{
    if (LIKELY(globalObject->worldIsNormal())) {
        if (auto* wrapper = node.wrapper())
            return wrapper;
    } else {
        if (auto* wrapper = getOutOfLineCachedWrapper(globalObject, node))
            return wrapper;
    }

    return createWrapper(lexicalGlobalObject, globalObject, node);
}

// In the C++ DOM, a node tree survives as long as there is a reference to its
// root. In the JavaScript DOM, a node tree survives as long as there is a
// reference to any node in the tree. To model the JavaScript DOM on top of
// the C++ DOM, we ensure that the root of every tree has a JavaScript wrapper.
void willCreatePossiblyOrphanedTreeByRemovalSlowCase(Node& root);
inline void willCreatePossiblyOrphanedTreeByRemoval(Node& root)
{
    if (!root.wrapper() && root.hasChildNodes())
        willCreatePossiblyOrphanedTreeByRemovalSlowCase(root);
}

inline WebCoreOpaqueRoot root(Node&);
inline WebCoreOpaqueRoot root(Node*);
inline WebCoreOpaqueRoot root(Document*);

ALWAYS_INLINE JSC::JSValue JSNode::nodeType(JSC::JSGlobalObject&) const
{
    return JSC::jsNumber(static_cast<uint8_t>(type()) & JSNodeTypeMask);
}

} // namespace WebCore
