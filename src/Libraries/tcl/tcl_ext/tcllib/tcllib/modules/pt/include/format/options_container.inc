/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

[list_begin options]
[include options_std.inc]

[opt_def -mode [const bulk]|[const incremental]]

The value of this option controls which methods of

[package pt::peg::container] instances are used to specify the
grammar, i.e. preload it into the container. There are two legal
values, as listed below. The default is [const bulk].

[list_begin definitions]
[def [const bulk]]

In this mode the methods [method start], [method add], [method modes],
and [method rules] are used to specify the grammar in a bulk manner,
i.e. as a set of nonterminal symbols, and two dictionaries mapping
from the symbols to their semantic modes and parsing expressions.

[para]

This mode is the default.

[def [const incremental]]

In this mode the methods [method start], [method add], [method mode],
and [method rule] are used to specify the grammar piecemal, with each
nonterminal having its own block of defining commands.

[list_end]

[opt_def -template string]

The value of this option is a string into which to put the generated
code and the other configuration settings. The various locations for
user-data are expected to be specified with the placeholders listed
below. The default value is "[const @code@]".

[list_begin definitions]

[def [const @user@]]
To be replaced with the value of the option [option -user].

[def [const @format@]]
To be replaced with the the constant [const CONTAINER].

[def [const @file@]]
To be replaced with the value of the option [option -file].

[def [const @name@]]
To be replaced with the value of the option [option -name].

[def [const @mode@]]
To be replaced with the value of the option [option -mode].

[def [const @code@]]
To be replaced with the generated code.

[list_end]
[list_end]
