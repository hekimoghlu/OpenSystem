/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

#include "ScriptWrappable.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Node;
class ScriptExecutionContext;

class NodeList : public ScriptWrappable, public RefCounted<NodeList> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(NodeList, WEBCORE_EXPORT);
public:
    virtual ~NodeList() = default;

    // DOM methods & attributes for NodeList
    virtual unsigned length() const = 0;
    virtual Node* item(unsigned index) const = 0;
    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }

    class Iterator {
    public:
        explicit Iterator(NodeList& list) : m_list(list) { }
        Node* next() { return m_list->item(m_index++); }

    private:
        size_t m_index { 0 };
        Ref<NodeList> m_list;
    };
    Iterator createIterator(ScriptExecutionContext*) { return Iterator(*this); }

    // Other methods (not part of DOM)
    virtual bool isLiveNodeList() const { return false; }
    virtual bool isChildNodeList() const { return false; }
    virtual bool isEmptyNodeList() const { return false; }
    virtual size_t memoryCost() const { return 0; }
};

} // namespace WebCore
