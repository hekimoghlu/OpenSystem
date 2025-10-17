/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#if DEVELOPMENT || DEBUG

#include <kern/startup.h>
#include <kern/zalloc.h>
#include <sys/proc_ro.h>
#include <sys/vm.h>

#include <tests/ktest.h>

#include <libkern/tree.h>

/*
 * RB tree node that we use for testing.
 * The nodes are compared using the numeric `rbt_id'.
 */
struct rbt_test_node {
	RB_ENTRY(rbt_test_node) link;
	unsigned int rbt_id; /* comparison id */
	unsigned int rbt_flags;
#define RBT_FLAG_IN_USE 1
#define RBT_MASK_IN_USE 1
};

typedef struct rbt_test_node * __single rbt_test_node_t;

/*
 * Comparison function for rbt test nodes.
 */
static int
rbt_cmp(struct rbt_test_node *a, struct rbt_test_node *b)
{
	if (!a && !b) {
		return 0;
	} else if (a && !b) {
		return -1;
	} else if (!a && b) {
		return 1;
	} else if (a->rbt_id == b->rbt_id) {
		return 0;
	} else if (a->rbt_id < b->rbt_id) {
		return -1;
	} else {
		return 1;
	}
}

/*
 * Define a red-black tree type we are going to test.
 */
RB_HEAD(_rb_test_tree, rbt_test_node);
RB_PROTOTYPE(_rb_test_tree, rbt_test_node, link, rbt_cmp)
RB_GENERATE(_rb_test_tree, rbt_test_node, link, rbt_cmp)

/*
 * Array of test nodes that we are going to use.
 */
#define RBT_TEST_NODE_COUNT 7
static struct rbt_test_node test_nodes[RBT_TEST_NODE_COUNT];
static int test_node_ids[RBT_TEST_NODE_COUNT] = {88, 66, 44, 22, 0, 77, 55 };


static size_t
rb_tree_insert_nodes(struct _rb_test_tree *tree, size_t count)
{
	unsigned int idx = 0;
	if (RBT_TEST_NODE_COUNT < count) {
		count = RBT_TEST_NODE_COUNT;
	}

	for (idx = 0; idx < count; idx++) {
		rbt_test_node_t node = &test_nodes[idx];
		T_EXPECT_EQ_INT(node->rbt_flags & RBT_MASK_IN_USE, 0, "Trying to insert a tree node that is already in use");
		node->rbt_id = test_node_ids[idx];
		RB_INSERT(_rb_test_tree, tree, node);
		node->rbt_flags |= RBT_FLAG_IN_USE;
	}
	return count;
}

static size_t
rb_tree_remove_nodes(struct _rb_test_tree *tree, size_t count)
{
	unsigned int idx = 0;
	if (RBT_TEST_NODE_COUNT < count) {
		count = RBT_TEST_NODE_COUNT;
	}

	for (idx = 0; idx < count; idx++) {
		rbt_test_node_t node = &test_nodes[idx];
		T_EXPECT_EQ_INT(node->rbt_flags & RBT_MASK_IN_USE, 1, "Trying to remove a tree node that is not in use");
		T_EXPECT_EQ_INT(node->rbt_id, test_node_ids[idx], "The node id does not match the node index");
		RB_REMOVE(_rb_test_tree, tree, node);
		node->rbt_flags &= ~RBT_FLAG_IN_USE;
	}
	return count;
}

static int
rb_tree_test_run(__unused int64_t in, int64_t *out)
{
	struct _rb_test_tree test_tree;
	RB_INIT(&test_tree);

	rb_tree_insert_nodes(&test_tree, 7);
	rb_tree_remove_nodes(&test_tree, 7);

	*out = 0;
	return 0;
}

SYSCTL_TEST_REGISTER(rb_tree_test, rb_tree_test_run);

#endif /* DEVELOPMENT || DEBUG */
