/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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


[section {Tcl/PARAM code representation of parsing expression grammars}]

The Tcl/PARAM representation of parsing expression grammars is Tcl
code whose execution will parse input per the grammar. The code is
based on the virtual machine documented in the
[manpage {PackRat Machine Specification}], using its instructions
and a few more to handle control flow.

[para]

Note that the generated code by itself is not functional. It expects
to be embedded into a framework which provides services like the PARAM
state, implementations for the PARAM instructions, etc.

The bulk of such a framework has to be specified through the option
[option -template]. The additional options

[list_begin options]
[opt_def -indent integer]
[opt_def -main string]
[opt_def -namespace string]
[opt_def -prelude string]
[opt_def -proc-command string]
[opt_def -runtime-command string]
[opt_def -self-command string]
[list_end]

provide code snippets which help to glue framework and generated code
together. Their placeholders are in the [emph generated] code.
