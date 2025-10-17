/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
	 Description of the concepts used in parsing expression
	 grammars and how their relate to each other. This is useful
	 to understand the chosen serialization.
}]
[list_begin enumerated]
[enum]

A [term {parsing expression grammar}] consists of a
[term {start parsing expression}] and a (possibly empty) list of
[term rules].

[enum]
Each rule defines a nonterminal symbol of the grammar, with its name,
semantic mode, and sentennial structure. The latter is provided by a
[term {parsing expression}].

[enum]
Each nonterminal symbol is specified at most once, with its name as
the identifying part.

[list_end]
