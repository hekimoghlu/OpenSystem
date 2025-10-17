/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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


The named directions are commands which tell the layout system in
which direction to go when placing the next element without an
explicit position specification.

They can also be used as arguments to the attribute [cmd then], and
the command [cmd by] for relative points, see there for the relevant
syntax.

[para]

The diagram core defines the directions of the compass rose, plus a
number of aliases. See below for the full list.

[para][image figure-27-corners-closed][para]

This overlaps with the pre-defined corners for closed elements. This
is used by the layout system, when are going in direction X the name
of the opposite direction is the name of the corner at which the new
element will be attached to the current position, and if this corner
does not exist the nearest actual corner by angle is used.

[para]

[list_begin definitions]
[call [cmd west]]
[call [cmd w]]
[call [cmd left]]
[call [cmd l]]

[call [cmd south]]
[call [cmd s]]
[call [cmd down]]
[call [cmd bottom]]
[call [cmd bot]]
[call [cmd b]]

[call [cmd east]]
[call [cmd e]]
[call [cmd right]]
[call [cmd r]]

[call [cmd north]]
[call [cmd n]]
[call [cmd up]]
[call [cmd top]]
[call [cmd t]]

[call [cmd northwest]]
[call [cmd nw]]
[call [cmd up-left]]
[call [cmd upleft]]
[call [cmd leftup]]

[call [cmd northeast]]
[call [cmd ne]]
[call [cmd up-right]]
[call [cmd upright]]
[call [cmd rightup]]

[call [cmd southwest]]
[call [cmd sw]]
[call [cmd down-left]]
[call [cmd downleft]]
[call [cmd leftdown]]

[call [cmd southeast]]
[call [cmd se]]
[call [cmd down-right]]
[call [cmd downright]]
[call [cmd rightdown]]
[list_end]
