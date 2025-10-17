/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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


[section {C/PARAM code representation of parsing expression grammars}]

[include whatis_cparam_rawc.inc]
[para]

The bulk of such a framework has to be specified through the option
[option -template]. The additional options

[list_begin options]
[opt_def -fun-qualifier string]
[opt_def -indent integer]
[opt_def -main string]
[opt_def -namespace string]
[opt_def -prelude string]
[opt_def -self-command string]
[opt_def -state-decl string]
[opt_def -state-ref string]
[opt_def -string-varname string]
[list_end]

provide code snippets which help to glue framework and generated code
together. Their placeholders are in the [emph generated] code.

[subsection Example]

We are forgoing an example of this representation, with apologies.
It would be way to large for this document.
