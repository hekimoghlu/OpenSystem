/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
/* This code is based on WebKit's wtf/RedBlackTree.h. */

#ifndef PAS_RED_BLACK_TREE_H
#define PAS_RED_BLACK_TREE_H

#include "pas_compact_atomic_ptr.h"
#include "pas_compact_tagged_atomic_ptr.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_red_black_tree_color {
    pas_red_black_tree_color_red,
    pas_red_black_tree_color_black,
};

typedef enum pas_red_black_tree_color pas_red_black_tree_color;

struct pas_red_black_tree;
struct pas_red_black_tree_jettisoned_nodes;
struct pas_red_black_tree_node;
typedef struct pas_red_black_tree pas_red_black_tree;
typedef struct pas_red_black_tree_jettisoned_nodes pas_red_black_tree_jettisoned_nodes;
typedef struct pas_red_black_tree_node pas_red_black_tree_node;

PAS_DEFINE_COMPACT_ATOMIC_PTR(pas_red_black_tree_node, pas_red_black_tree_node_ptr);
PAS_DEFINE_COMPACT_TAGGED_ATOMIC_PTR(uintptr_t, pas_red_black_tree_node_tagged_ptr);

struct pas_red_black_tree {
    pas_red_black_tree_node_ptr root;
};

struct pas_red_black_tree_node {
    pas_red_black_tree_node_ptr left;
    pas_red_black_tree_node_ptr right;
    pas_red_black_tree_node_tagged_ptr parent;
};

/* This struct enables the tree to be iterated by the heap enumerator at any time. This is never read by
   the red-black tree algorithm itself. For example, if you don't want your tree to be heap-enumerated,
   then you can just pass a dummy stack-allocated or even global struct here. Passing a global would be
   fine in that case except that you'd potentially create write contention. */
struct pas_red_black_tree_jettisoned_nodes {
    pas_red_black_tree_node* first_rotate_jettisoned;
    pas_red_black_tree_node* second_rotate_jettisoned;
    pas_red_black_tree_node* remove_jettisoned;
};

#define PAS_RED_BLACK_TREE_INITIALIZER { .root = PAS_COMPACT_ATOMIC_PTR_INITIALIZER }

#if PAS_ENABLE_TESTING
PAS_API extern void (*pas_red_black_tree_validate_enumerable_callback)(void);

static inline void pas_red_black_tree_validate_enumerable(void)
{
    if (pas_red_black_tree_validate_enumerable_callback)
        pas_red_black_tree_validate_enumerable_callback();
}
#else /* PAS_ENABLE_TESTING -> so !PAS_ENABLE_TESTING */
static inline void pas_red_black_tree_validate_enumerable(void)
{
}
#endif /* PAS_ENABLE_TESTING -> so end of !PAS_ENABLE_TESTING */

static inline pas_red_black_tree_node* pas_red_black_tree_get_root(pas_red_black_tree* tree)
{
    return pas_red_black_tree_node_ptr_load(&tree->root);
}

static inline pas_red_black_tree_node* pas_red_black_tree_node_get_left(pas_red_black_tree_node* node)
{
    return pas_red_black_tree_node_ptr_load(&node->left);
}

static inline pas_red_black_tree_node* pas_red_black_tree_node_get_right(pas_red_black_tree_node* node)
{
    return pas_red_black_tree_node_ptr_load(&node->right);
}

static inline pas_red_black_tree_node* pas_red_black_tree_node_get_parent(pas_red_black_tree_node* node)
{
    return (pas_red_black_tree_node*)(pas_red_black_tree_node_tagged_ptr_load(&node->parent) & ~1lu);
}

static inline pas_red_black_tree_color pas_red_black_tree_node_get_color(pas_red_black_tree_node* node)
{
    return (pas_red_black_tree_color)(pas_red_black_tree_node_tagged_ptr_load(&node->parent) & 1);
}

static inline void pas_red_black_tree_set_root(pas_red_black_tree* tree,
                                               pas_red_black_tree_node* value)
{
    pas_red_black_tree_validate_enumerable();
    pas_red_black_tree_node_ptr_store(&tree->root, value);
    pas_red_black_tree_validate_enumerable();
}

static inline void pas_red_black_tree_node_set_left(pas_red_black_tree_node* node,
                                                    pas_red_black_tree_node* value)
{
    pas_red_black_tree_validate_enumerable();
    pas_red_black_tree_node_ptr_store(&node->left, value);
    pas_red_black_tree_validate_enumerable();
}

static inline void pas_red_black_tree_node_set_right(pas_red_black_tree_node* node,
                                                     pas_red_black_tree_node* value)
{
    pas_red_black_tree_validate_enumerable();
    pas_red_black_tree_node_ptr_store(&node->right, value);
    pas_red_black_tree_validate_enumerable();
}

static inline void pas_red_black_tree_node_set_parent(pas_red_black_tree_node* node,
                                                      pas_red_black_tree_node* value)
{
    uintptr_t tagged_value = pas_red_black_tree_node_tagged_ptr_load(&node->parent);
    tagged_value = (tagged_value & 1) | (uintptr_t)value;
    pas_red_black_tree_validate_enumerable();
    pas_red_black_tree_node_tagged_ptr_store(&node->parent, tagged_value);
    pas_red_black_tree_validate_enumerable();
}

static inline void pas_red_black_tree_node_set_color(pas_red_black_tree_node* node,
                                                     pas_red_black_tree_color value)
{
    uintptr_t tagged_value = pas_red_black_tree_node_tagged_ptr_load(&node->parent);
    tagged_value = (tagged_value & ~1lu) | (uintptr_t)value;
    pas_red_black_tree_validate_enumerable();
    pas_red_black_tree_node_tagged_ptr_store(&node->parent, tagged_value);
    pas_red_black_tree_validate_enumerable();
}

typedef int (*pas_red_black_tree_node_compare_callback)(pas_red_black_tree_node* a,
                                                        pas_red_black_tree_node* b);
typedef int (*pas_red_black_tree_key_compare_callback)(pas_red_black_tree_node* node,
                                                       void* key);

static inline pas_red_black_tree_node*
pas_red_black_tree_node_minimum(pas_red_black_tree_node* node)
{
    for (;;) {
        pas_red_black_tree_node* left;
        left = pas_red_black_tree_node_get_left(node);
        if (!left)
            return node;
        node = left;
    }
}

static inline pas_red_black_tree_node*
pas_red_black_tree_node_maximum(pas_red_black_tree_node* node)
{
    for (;;) {
        pas_red_black_tree_node* right;
        right = pas_red_black_tree_node_get_right(node);
        if (!right)
            return node;
        node = right;
    }
}

static inline pas_red_black_tree_node*
pas_red_black_tree_node_successor(pas_red_black_tree_node* node)
{
    pas_red_black_tree_node* x;
    pas_red_black_tree_node* y;
    pas_red_black_tree_node* x_right;
    
    x = node;
    x_right = pas_red_black_tree_node_get_right(x);
    if (x_right)
        return pas_red_black_tree_node_minimum(x_right);
    
    y = pas_red_black_tree_node_get_parent(x);
    while (y && x == pas_red_black_tree_node_get_right(y)) {
        x = y;
        y = pas_red_black_tree_node_get_parent(y);
    }
    return y;
}

static inline pas_red_black_tree_node*
pas_red_black_tree_node_predecessor(pas_red_black_tree_node* node)
{
    pas_red_black_tree_node* x;
    pas_red_black_tree_node* y;
    pas_red_black_tree_node* x_left;
    
    x = node;
    x_left = pas_red_black_tree_node_get_left(x);
    if (x_left)
        return pas_red_black_tree_node_maximum(x_left);
    
    y = pas_red_black_tree_node_get_parent(x);
    while (y && x == pas_red_black_tree_node_get_left(y)) {
        x = y;
        y = pas_red_black_tree_node_get_parent(y);
    }
    return y;
}

static inline void pas_red_black_tree_node_reset(pas_red_black_tree_node* node)
{
    pas_red_black_tree_node_set_left(node, NULL);
    pas_red_black_tree_node_set_right(node, NULL);
    pas_red_black_tree_node_set_parent(node, NULL);
    pas_red_black_tree_node_set_color(node, pas_red_black_tree_color_red);
}

static inline void pas_red_black_tree_construct(pas_red_black_tree* tree)
{
    pas_red_black_tree_set_root(tree, NULL);
}

PAS_API void pas_red_black_tree_insert(pas_red_black_tree* tree,
                                       pas_red_black_tree_node* node,
                                       pas_red_black_tree_node_compare_callback compare_callback,
                                       pas_red_black_tree_jettisoned_nodes* jettisoned_nodes);

PAS_API pas_red_black_tree_node*
pas_red_black_tree_remove(pas_red_black_tree* tree,
                          pas_red_black_tree_node* node,
                          pas_red_black_tree_jettisoned_nodes* jettisoned_nodes);

static inline pas_red_black_tree_node*
pas_red_black_tree_node_find_exact(pas_red_black_tree_node* root,
                                   void* key,
                                   pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* current;
    for (current = root; current;) {
        int compare_result;
        
        compare_result = compare_callback(current, key);
        
        if (compare_result == 0)
            return current;
        if (compare_result > 0)
            current = pas_red_black_tree_node_get_left(current);
        else
            current = pas_red_black_tree_node_get_right(current);
    }
    return NULL;
}

static inline pas_red_black_tree_node*
pas_red_black_tree_find_exact(pas_red_black_tree* tree,
                              void* key,
                              pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* root;
    root = pas_red_black_tree_get_root(tree);
    if (!root)
        return NULL;
    return pas_red_black_tree_node_find_exact(root, key, compare_callback);
}

static inline pas_red_black_tree_node*
pas_red_black_tree_node_find_least_greater_than_or_equal(
    pas_red_black_tree_node* root,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* current;
    pas_red_black_tree_node* best;
    
    best = NULL;
    
    for (current = root; current;) {
        int compare_result;
        
        compare_result = compare_callback(current, key);
        
        if (compare_result == 0)
            return current;
        
        if (compare_result < 0)
            current = pas_red_black_tree_node_get_right(current);
        else {
            best = current;
            current = pas_red_black_tree_node_get_left(current);
        }
    }
    
    return best;
}

static inline pas_red_black_tree_node*
pas_red_black_tree_find_least_greater_than_or_equal(
    pas_red_black_tree* tree,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* root;
    root = pas_red_black_tree_get_root(tree);
    if (!root)
        return NULL;
    return pas_red_black_tree_node_find_least_greater_than_or_equal(root, key, compare_callback);
}

static inline pas_red_black_tree_node*
pas_red_black_tree_node_find_least_greater_than(
    pas_red_black_tree_node* root,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* current;
    pas_red_black_tree_node* best;
    
    best = NULL;
    
    for (current = root; current;) {
        int compare_result;
        
        compare_result = compare_callback(current, key);
        
        if (compare_result <= 0)
            current = pas_red_black_tree_node_get_right(current);
        else {
            best = current;
            current = pas_red_black_tree_node_get_left(current);
        }
    }
    
    return best;
}

static inline pas_red_black_tree_node*
pas_red_black_tree_find_least_greater_than(
    pas_red_black_tree* tree,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* root;
    root = pas_red_black_tree_get_root(tree);
    if (!root)
        return NULL;
    return pas_red_black_tree_node_find_least_greater_than(root, key, compare_callback);
}

static inline pas_red_black_tree_node*
pas_red_black_tree_node_find_greatest_less_than_or_equal(
    pas_red_black_tree_node* root,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* current;
    pas_red_black_tree_node* best;
    
    best = NULL;
    
    for (current = root; current;) {
        int compare_result;
        
        compare_result = compare_callback(current, key);
        
        if (compare_result == 0)
            return current;
        
        if (compare_result > 0)
            current = pas_red_black_tree_node_get_left(current);
        else {
            best = current;
            current = pas_red_black_tree_node_get_right(current);
        }
    }
    
    return best;
}

static inline pas_red_black_tree_node*
pas_red_black_tree_find_greatest_less_than_or_equal(
    pas_red_black_tree* tree,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* root;
    root = pas_red_black_tree_get_root(tree);
    if (!root)
        return NULL;
    return pas_red_black_tree_node_find_greatest_less_than_or_equal(root, key, compare_callback);
}

static inline pas_red_black_tree_node*
pas_red_black_tree_node_find_greatest_less_than(
    pas_red_black_tree_node* root,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* current;
    pas_red_black_tree_node* best;
    
    best = NULL;
    
    for (current = root; current;) {
        int compare_result;
        
        compare_result = compare_callback(current, key);
        
        if (compare_result >= 0)
            current = pas_red_black_tree_node_get_left(current);
        else {
            best = current;
            current = pas_red_black_tree_node_get_right(current);
        }
    }
    
    return best;
}

static inline pas_red_black_tree_node*
pas_red_black_tree_find_greatest_less_than(
    pas_red_black_tree* tree,
    void* key,
    pas_red_black_tree_key_compare_callback compare_callback)
{
    pas_red_black_tree_node* root;
    root = pas_red_black_tree_get_root(tree);
    if (!root)
        return NULL;
    return pas_red_black_tree_node_find_greatest_less_than(root, key, compare_callback);
}

static inline pas_red_black_tree_node* pas_red_black_tree_minimum(pas_red_black_tree* tree)
{
    pas_red_black_tree_node* root;
    root = pas_red_black_tree_get_root(tree);
    if (!root)
        return NULL;
    return pas_red_black_tree_node_minimum(root);
}

static inline pas_red_black_tree_node* pas_red_black_tree_maximum(pas_red_black_tree* tree)
{
    pas_red_black_tree_node* root;
    root = pas_red_black_tree_get_root(tree);
    if (!root)
        return NULL;
    return pas_red_black_tree_node_maximum(root);
}

/* This is a O(n) operation. */
PAS_API size_t pas_red_black_tree_size(pas_red_black_tree* tree);

static inline bool pas_red_black_tree_is_empty(pas_red_black_tree* tree)
{
    return !pas_red_black_tree_get_root(tree);
}

PAS_END_EXTERN_C;

#endif /* PAS_RED_BLACK_TREE_H */

