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

namespace JSC {

class ProfileTreeNode {
    typedef UncheckedKeyHashMap<String, ProfileTreeNode> Map;
    typedef Map::KeyValuePairType MapEntry;

public:
    ProfileTreeNode()
        : m_count(0)
        , m_children(0)
    {
    }

    ~ProfileTreeNode()
    {
        delete m_children;
    }

    ProfileTreeNode* sampleChild(const char* name)
    {
        if (!m_children)
            m_children = new Map();
    
        ProfileTreeNode newEntry;
        Map::AddResult result = m_children->add(String(name), newEntry);
        ProfileTreeNode* childInMap = &result.iterator->value;
        ++childInMap->m_count;
        return childInMap;
    }

    void dump()
    {
        dumpInternal(0);
    }

    uint64_t count()
    {
        return m_count;
    }

    uint64_t childCount()
    {
        if (!m_children)
            return 0;
        uint64_t childCount = 0;
        for (Map::iterator it = m_children->begin(); it != m_children->end(); ++it)
            childCount += it->value.count();
        return childCount;
    }
    
private:
    void dumpInternal(unsigned indent)
    {
        if (!m_children)
            return;

        // Copy pointers to all children into a vector, and sort the vector by sample count.
        Vector<MapEntry*> entries;
        for (Map::iterator it = m_children->begin(); it != m_children->end(); ++it)
            entries.append(&*it);
        qsort(entries.begin(), entries.size(), sizeof(MapEntry*), compareEntries);

        // Iterate over the children in sample-frequency order.
        for (auto* entry : entries) {
            // Print the number of samples, the name of this node, and the number of samples that are stack-top
            // in this node (samples directly within this node, excluding samples in children.
            for (unsigned i = 0; i < indent; ++i)
                dataLogF("    ");
            dataLogF("% 8lld: %s (%lld stack top)\n",
                static_cast<long long>(entry->value.count()),
                entry->key.utf8().data(),
                static_cast<long long>(entry->value.count() - entry->value.childCount()));

            // Recursively dump the child nodes.
            entry->value.dumpInternal(indent + 1);
        }
    }

    static int compareEntries(const void* a, const void* b)
    {
        uint64_t da = (*static_cast<MapEntry* const *>(a))->value.count();
        uint64_t db = (*static_cast<MapEntry* const *>(b))->value.count();
        return (da < db) - (da > db);
    }

    uint64_t m_count;
    Map* m_children;
};

} // namespace JSC
