/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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


In this section we are working a complete example, starting with a PEG
grammar and ending with running the parser generated from it over some
input, following the outline shown in the figure below:

[para][image flow][para]

Our grammar, assumed to the stored in the file [file calculator.peg]
is

[include expr_peg.inc]

From this we create a snit-based parser

[include full_[vset MODE].inc]

which leaves us with the parser package and class written to the file
[file calculator.tcl].

Assuming that this package is then properly installed in a place where
Tcl can find it we can now use this class via a script like

[include parser_use.inc]

where the abstract syntax tree stored in the variable will look like

[para][include expr_ast.inc][para]

assuming that the input file and channel contained the text

[example { 120+5 }]

A more graphical representation of the tree would be

[para][image expr_ast][para]

Regardless, at this point it is the user's responsibility to work with
the tree to reach whatever goal she desires. I.e. analyze it,
transform it, etc. The package [package pt::ast] should be of help
here, providing commands to walk such ASTs structures in various ways.

[para]

One important thing to note is that the parsers used here return a
data structure representing the structure of the input per the grammar
underlying the parser. There are [emph no] callbacks during the
parsing process, i.e. no [term {parsing actions}], as most other
parsers will have.

[para]

Going back to the last snippet of code, the execution of the parser
for some input, note how the parser instance follows the specified
[term {Parser API}].
