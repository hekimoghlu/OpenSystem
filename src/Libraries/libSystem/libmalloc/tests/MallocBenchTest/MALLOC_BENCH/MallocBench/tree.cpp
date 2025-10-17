/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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
#include "tree.h"
#include <limits>
#include <stdlib.h>
#include <strings.h>

#include "mbmalloc.h"

namespace {

struct Node {
    void* operator new(size_t size)
    {
        return mbmalloc(size);
    }

    void operator delete(void* p, size_t size)
    {
        mbfree(p, size);
    }

    Node(Node* left, Node* right, size_t payloadSize, size_t id)
        : m_refCount(1)
        , m_left(left)
        , m_right(right)
        , m_payload(static_cast<char*>(mbmalloc(payloadSize)))
        , m_payloadSize(payloadSize)
        , m_id(id)
    {
        if (m_left)
            m_left->ref();
        if (m_right)
            m_right->ref();
        bzero(m_payload, payloadSize);
    }

    ~Node()
    {
        if (m_left)
            m_left->deref();
        if (m_right)
            m_right->deref();
        mbfree(m_payload, m_payloadSize);
    }

    void ref()
    {
        ++m_refCount;
    }

    void deref()
    {
        if (m_refCount == 1)
            delete this;
        else
            --m_refCount;
    }
    
    size_t id() { return m_id; }
    Node* left() { return m_left; }
    Node* right() { return m_right; }

    void setLeft(Node* left)
    {
        left->ref();
        if (m_left)
            m_left->deref();
        
        m_left = left;
    }

    void setRight(Node* right)
    {
        right->ref();
        if (m_right)
            m_right->deref();
        
        m_right = right;
    }

    unsigned m_refCount;
    Node* m_left;
    Node* m_right;
    char* m_payload;
    size_t m_payloadSize;
    size_t m_id;
};

void verify(Node* node, Node* left, Node* right)
{
    if (left && left->id() >= node->id())
        abort();

    if (right && right->id() <= node->id())
        abort();
}

Node* createTree(size_t depth, size_t& counter)
{
    if (!depth)
        return 0;

    Node* left = createTree(depth - 1, counter);
    size_t id = counter++;
    Node* right = createTree(depth - 1, counter);

    Node* result = new Node(left, right, ((depth * 8) & (64 - 1)) | 1, id);

    verify(result, left, right);

    if (left)
        left->deref();
    if (right)
        right->deref();
    return result;
}

Node* createTree(size_t depth)
{
    size_t counter = 0;
    return createTree(depth, counter);
}

void churnTree(Node* node, size_t stride, size_t& counter)
{
    if (!node)
        return;
    
    churnTree(node->left(), stride, counter);

    if (node->left() && !(counter % stride)) {
        Node* left = new Node(node->left()->left(), node->left()->right(), (counter & (64 - 1)) | 1, node->left()->id());
        Node* right = new Node(node->right()->left(), node->right()->right(), (counter & (64 - 1)) | 1, node->right()->id());
        node->setLeft(left);
        node->setRight(right);
        left->deref();
        right->deref();
    }
    ++counter;

    churnTree(node->right(), stride, counter);

    verify(node, node->left(), node->right());
}

void churnTree(Node* tree, size_t stride)
{
    size_t counter;
    churnTree(tree, stride, counter);
}

} // namespace

void benchmark_tree_allocate(CommandLine& commandLine)
{
    size_t times = 24;
    size_t depth = 16;
    if (commandLine.isParallel()) {
        times *= 4;
        depth = 13;
    }

    for (size_t time = 0; time < times; ++time) {
        Node* tree = createTree(depth);
        tree->deref();
    }
}

void benchmark_tree_traverse(CommandLine& commandLine)
{
    size_t times = 256;
    size_t depth = 15;
    if (commandLine.isParallel()) {
        times = 512;
        depth = 13;
    }

    Node* tree = createTree(depth);
    for (size_t time = 0; time < times; ++time)
        churnTree(tree, std::numeric_limits<size_t>::max()); // Reuse this to iterate and validate.
    tree->deref();
}

void benchmark_tree_churn(CommandLine& commandLine)
{
    size_t times = 130;
    size_t depth = 15;
    if (commandLine.isParallel()) {
        times *= 4;
        depth = 12;
    }

    Node* tree = createTree(depth);
    for (size_t time = 0; time < times; ++time)
        churnTree(tree, 8);
    tree->deref();
}
