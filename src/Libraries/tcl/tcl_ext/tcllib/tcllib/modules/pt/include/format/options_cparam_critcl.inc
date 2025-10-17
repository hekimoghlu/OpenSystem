/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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

[opt_def -class string]

The value of this option is the name of the class to generate, without
leading colons.

The default value is [const CLASS].

[para]

For a simple value [var X] without colons, like CLASS, the parser
command will be [var X]::[var X]. Whereas for a namespaced value
[var X::Y] the parser command will be [var X::Y].


[opt_def -package string]

The value of this option is the name of the package to generate.

The default value is [const PACKAGE].

[list_end]
