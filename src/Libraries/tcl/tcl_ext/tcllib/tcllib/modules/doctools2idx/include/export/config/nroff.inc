/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

The nroff export plugin recognizes the following configuration
variables and changes its behaviour as they specify.

[list_begin arguments]

[arg_def string user]
This standard configuration variable contains the name of the user
running the process which invoked the export plugin.

The plugin puts this information into the provenance comment at the
beginning of the generated document.

[arg_def string file]

This standard configuration variable contains the name of the file the
index came from. This variable may not be set or contain the empty
string.

The plugin puts this information, if defined, i.e. set and not the
empty string, into the provenance comment at the beginning of the
generated document.


[arg_def boolean inline]

If this flag is set (default) the plugin will place the definitions of
the man macro set directly into the output.

[para]

If this flag is not set, the plugin will place a reference to the
definitions of the man macro set into the output, but not the macro
definitions themselves.

[list_end]

[emph Note] that this plugin ignores the standard configuration
variables [var format], and [var map], and their values.
