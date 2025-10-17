/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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


[call [arg objectName] [method destroy]]

This method destroys the parser instance, releasing all claimed memory
and other resources, and deleting the instance command.

[para]

The result of the command is the empty string.


[call [arg objectName] [method parse] [arg chan]]

This method runs the parser using the contents of [arg chan] as input
(starting at the current location in the channel), until parsing is
not possible anymore, either because parsing has completed, or run
into a syntax error.

[include channel_notes.inc]

[para]

Upon successful completion the command returns an abstract syntax tree
as its result.

This AST is in the form specified in section
[sectref {AST serialization format}].

As a plain nested Tcl-list it can then be processed with any Tcl
commands the user likes, doing transformations, semantic checks, etc.

To help in this the package [package pt::ast] provides a set of
convenience commands for validation of the tree's basic structure,
printing it for debugging, and walking it either from the bottom up,
or top down.

[para]

When encountering a syntax error the command will throw an error instead.

This error will be a 4-element Tcl-list, containing, in the order
listed below:

[list_begin enumerated]
[enum]
The string [const pt::rde] identifying it as parser runtime error.

[enum]
The location of the parse error, as character offset from the
beginning of the parsed input.

[enum]
The location of parse error, now as a 2-element list containing
line-number and column in the line.

[enum]
A set of atomic parsing expressions indicating encoding the characters
and/or nonterminal symbols the parser expected to see at the location
of the parse error, but did not get.

 For the specification of atomic parsing expressions please see the
section [sectref {PE serialization format}].

[list_end]


[call [arg objectName] [method parset] [arg text]]

This method runs the parser using the string in [arg text] as input.
In all other ways it behaves like the method [method parse], shown
above.
