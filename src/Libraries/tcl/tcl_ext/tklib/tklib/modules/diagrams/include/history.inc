/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

[call [cmd [var n]th] [opt [arg corner]]]

This command asks the diagram history for the [var n]th element
created, searching from the beginning of the history (counting from 1)
and returns it as its result.

If the [arg corner] is specified then the value for this corner is
returned instead.


[call [cmd [var n]th] [cmd last] [opt [arg corner]]]

This command asks the diagram history for the [var n]th element
created, searching from the end of the history and returns it as its
result.

If the [arg corner] is specified then the value for this corner is
returned instead.


[call [cmd [var n]th] [arg shape] [opt [arg corner]]]

This command asks the diagram history for the [var n]th element
created, of the given [arg shape], searching from the beginning of the
history (counting from 1) and returns it as its result.

If the [arg corner] is specified then the value for this corner is
returned instead.


[call [cmd [var n]th] [cmd last] [arg shape] [opt [arg corner]]]

This command asks the diagram history for the [var n]th element
created, of the given [arg shape], searching from the end of the
history and returns it as its result.

If the [arg corner] is specified then the value for this corner is
returned instead.


[call [cmd last] [opt [arg corner]]]
[call [cmd last] [arg shape] [opt [arg corner]]]

Convenience commands mapping to "[cmd {1st last}]"
and "[cmd {1st last}] [arg shape]".


[call [cmd 1st]]
[call [cmd 2nd]]
[call [cmd 3rd]]

Aliases for [cmd 1th], [cmd 2th], and [cmd 3th], for readability,
usable whereever [cmd [var n]th] can ocur.
