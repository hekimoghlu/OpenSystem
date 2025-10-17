/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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

The text export plugin recognizes the following configuration
variables and changes its behaviour as they specify.

[list_begin arguments]

[arg_def dictionary map]

This standard configuration variable contains a dictionary mapping
from the symbolic files names in manpage references to the actual
filenames and/or urls to be used in the output.

[para]

Url references and symbolic file names without a mapping are used
unchanged.

[list_end]

[emph Note] that this plugin ignores the standard configuration
variables [var user], [var file], and [var format], and their values.
