/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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

[def "[cmd arrowhead] [arg spec]"]

[image figure-19-style-arrowheads]
Specifies where to draw arrowheads on the [cmd line] element, at the
beginning or end, at both ends, or none.

If not specified the system falls back to the value taken from the
language variable [var arrowhead], which itself defaults to
[const none].

[include ../values/arrowhead.inc]

Note that the values "start", "end", "-", "->", "<-", and "<->" are
all accepted as shorthands for the [cmd arrowhead] command using them
as argument.
