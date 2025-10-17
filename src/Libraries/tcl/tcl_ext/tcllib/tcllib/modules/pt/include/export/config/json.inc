/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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

[section Configuration]

The JSON export plugin recognizes the following configuration
variables and changes its behaviour as they specify.

[list_begin arguments]
[arg_def boolean indented]

If this flag is set the plugin will break the generated JSON code
across lines and indent it according to its inner structure, with each
key of a dictionary on a separate line.

[para]

If this flag is not set (the default), the whole JSON object will be
written on a single line, with minimum spacing between all elements.


[arg_def boolean aligned]

If this flag is set the generator ensures that the values for the keys
in a dictionary are vertically aligned with each other, for a nice
table effect. To make this work this also implies that [var indented]
is set.

[para]

If this flag is not set (the default), the output is formatted as per
the value of [var indented], without trying to align the values for
dictionary keys.

[list_end]

[emph Note] that this plugin ignores the standard configuration
variables [var user], [var format], [var file], and [var name], and
their values.
