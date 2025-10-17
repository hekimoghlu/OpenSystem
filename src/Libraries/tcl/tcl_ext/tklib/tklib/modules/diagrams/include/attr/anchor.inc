/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

[def "[cmd anchor] [arg name]"]
[def [cmd ljust]]
[def [cmd rjust]]
[def [cmd above]]
[def [cmd below]]

[image figure-22-text-anchoring-3]
Specifies the anchor of the text which is to be placed at the
element's center, by name. I.e. this attribute defines the text's
position relative to the element's center.

The value is ignored if no text was specified for the element.

If not specified the system falls back to the value taken from the
language variable [var anchor], which itself defaults to
[const center].

[include ../values/anchor.inc]

The commands without arguments are all shorthands with the anchor
implied. Note that they do not combine, only the last is used. For
comined directions the main attribute command, [cmd anchor] has to be
used.

[list_begin definitions]
[def [cmd ljust]]	"anchor west"
[def [cmd rjust]]	"anchor east"
[def [cmd above]]	"anchor south"
[def [cmd below]]	"anchor north"
[list_end]