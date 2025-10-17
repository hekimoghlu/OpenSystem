/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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

[def "[cmd style] [arg spec]"]

[image figure-18-style-dash]
Specifies the style used to draw the lines of the element.

If not specified the system falls back to the value taken from the
language variable [var linestyle], which itself defaults to
[const solid] lines.

[include ../values/style.inc]

Note that the values "solid", "dot(ted)", "dash(ed)", "dash-dot", and
"dash-dot-dot" are all accepted as shorthands for the [cmd style]
command using them as argument.
