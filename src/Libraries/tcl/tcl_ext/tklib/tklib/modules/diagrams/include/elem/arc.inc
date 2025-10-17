/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

[call [cmd arc] [arg attr]...]

[image figure-02-arc]
[include cat_open.inc]

Note however that it also has the compass rose of closed elements as
its corners, with the center of the arc's circle as the center of the
compass and the other points on the circle the arc is part of.

It handles the attributes

[list_begin definitions]
[include ../attr/anchor.inc]
[include ../attr/clockwise.inc]
[include ../attr/color.inc]
[include ../attr/counterclockwise.inc]
[include ../attr/fillcolor.inc]
[include ../attr/arc_from.inc]
[include ../attr/justify.inc]
[include ../attr/arc_radius.inc]
[include ../attr/stroke.inc]
[include ../attr/style.inc]
[include ../attr/text.inc]
[include ../attr/textcolor.inc]
[include ../attr/textfont.inc]
[include ../attr/arc_to.inc]
[list_end]
