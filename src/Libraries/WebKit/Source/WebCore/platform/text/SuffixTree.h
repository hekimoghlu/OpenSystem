/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class UnicodeCodebook {
public:
    static int codeWord(UChar c) { return c; }
    enum { codeSize = 1 << 8 * sizeof(UChar) };
};

class ASCIICodebook {
public:
    static int codeWord(UChar c) { return c & (codeSize - 1); }
    enum { codeSize = 1 << (8 * sizeof(char) - 1) };
};

template<typename Codebook>
class SuffixTree {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(SuffixTree);
    WTF_MAKE_NONCOPYABLE(SuffixTree)
public:
    SuffixTree(const String& text, unsigned depth)
        : m_depth(depth)
        , m_leaf(true)
    {
        build(text);
    }

    bool mightContain(const String& query)
    {
        Node* current = &m_root;
        int limit = std::min(m_depth, query.length());
        for (int i = 0; i < limit; ++i) {
            auto it = current->find(Codebook::codeWord(query[i]));
            if (it == current->end())
                return false;
            current = it->node;
        }
        return true;
    }

private:
    class Node {
        WTF_MAKE_TZONE_ALLOCATED_INLINE(Node);
    public:
        Node(bool isLeaf = false)
            : m_isLeaf(isLeaf)
        {
        }

        ~Node()
        {
            for (auto& entry : m_children) {
                auto* child = entry.node;
                if (child && !child->m_isLeaf)
                    delete child;
            }
        }

        Node*& childAt(int codeWord);

        auto find(int codeWord)
        {
            return std::find_if(m_children.begin(), m_children.end(), [codeWord](auto& entry) {
                return entry.codeWord == codeWord;
            });
        }

        auto end() { return m_children.end(); }

    private:
        struct ChildWithCodeWord {
            int codeWord;
            Node* node;
        };

        Vector<ChildWithCodeWord> m_children;
        bool m_isLeaf;
    };

    void build(const String& text)
    {
        for (unsigned base = 0; base < text.length(); ++base) {
            Node* current = &m_root;
            unsigned limit = std::min(base + m_depth, text.length());
            for (unsigned offset = 0; base + offset < limit; ++offset) {
                ASSERT(current != &m_leaf);
                Node*& child = current->childAt(Codebook::codeWord(text[base + offset]));
                if (!child)
                    child = base + offset + 1 == limit ? &m_leaf : new Node();
                current = child;
            }
        }
    }

    Node m_root;
    unsigned m_depth;

    // Instead of allocating a fresh empty leaf node for ever leaf in the tree
    // (there can be a lot of these), we alias all the leaves to this "static"
    // leaf node.
    Node m_leaf;
};

template<typename Codebook>
inline auto SuffixTree<Codebook>::Node::childAt(int codeWord) -> Node*&
{
    auto it = find(codeWord);
    if (it != m_children.end())
        return it->node;
    m_children.append(ChildWithCodeWord { codeWord, nullptr });
    return m_children.last().node;
}

} // namespace WebCore
