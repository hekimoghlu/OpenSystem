/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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


Corners are named values for in elements, usually locations.

[list_begin itemized]

[item]

The [term closed] elements define corners for the compass rose,
including the "center", and their "width" and "height".

[para][image figure-27-corners-closed][para]

[item]
[cmd block] elements additionally export all variables which were set
during their definition as corners.

[item]
The [term open] elements on the other hand define "start", "end", and
"center". The first two map to the locations originally provided
through the attributes [cmd from] and [cmd to] of the element.

[para][image figure-28-corners-open][para]

[item]
The center of [cmd line] and [cmd move] elements is the location
halfway between "start" and "end" corners, this is regardless of any
intermediate locations the element may have.

[item]
The [cmd line] and [cmd move] elements additionally name all their
locations as corners using numbers as names, starting from [const 1]
(equivalent to "start"), in order of traversal.

[para][image figure-15-spline-1][para]

[item]
The center of [cmd arc] elements is the center of the circle the arc
is part off.

[item]
The [cmd arc] elements additionally define the compass rose of
closed elements as well.

[list_end]
