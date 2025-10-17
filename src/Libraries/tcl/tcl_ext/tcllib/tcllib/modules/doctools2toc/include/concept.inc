/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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

[comment {
	 Description of the concepts used in tables of contents and how
	 their relate to each other. This is useful to understand the
	 serialization chosen for tables of contents.
}]
[list_begin enumerated]
[enum]

A [term {table of contents}] consists of a (possibly empty) list of
[term elements].

[enum]
Each element in the list is identified by its label.

[enum]
Each element is either a [term reference], or a [term division].

[enum]
Each reference has an associated document, identified by a symbolic
id, and a textual description.

[enum]
Each division may have an associated document, identified by a
symbolic id. 

[enum]
Each division consists consists of a (possibly empty) list of
[term elements], with each element following the rules as specified in
item 2 and above.

[list_end]

A few notes

[list_begin enumerated]
[enum]
The above rules span up a tree of elements, with references as the
leaf nodes, and divisions as the inner nodes, and each element
representing an entry in the whole table of contents.

[enum]
The identifying labels of any element E are unique within their
division (or toc), and the full label of any element E is the list of
labels for all nodes on the unique path from the root of the tree to
E, including E.

[list_end]
