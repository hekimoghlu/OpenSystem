/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#ifndef PAS_COMPACT_CARTESIAN_TREE_NODE_PTR_H
#define PAS_COMPACT_CARTESIAN_TREE_NODE_PTR_H

#include "pas_compact_ptr.h"

PAS_BEGIN_EXTERN_C;

struct pas_cartesian_tree_node;
typedef struct pas_cartesian_tree_node pas_cartesian_tree_node;

PAS_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
PAS_DEFINE_COMPACT_PTR(pas_cartesian_tree_node, pas_compact_cartesian_tree_node_ptr);
PAS_ALLOW_UNSAFE_BUFFER_USAGE_END

PAS_END_EXTERN_C;

#endif /* PAS_COMPACT_CARTESIAN_TREE_NODE_PTR_H */

