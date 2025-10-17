/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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

[def "[cmd then] [arg location]"]
[def "[cmd then] (<direction> [opt [arg length]])..."]
[def "(<direction> [opt [arg length]])..."]

This attribute specifies an intermediate location the [cmd line]
element has to go through.

It can be specified multiple times, with each use adding one
additional location to the series which the line will go
through. These location will be traversed in the order they were
specified.

[para]

The location can be given explicitly, or as a series of directions
with distances. In the latter case the names of all known directions
are accepted for the direction part.

If no distance is specified for a direction the system falls back to
the value taken from the language variable [var movelength], which
itself defaults to the pixel equivalent of [const {2 cm}].

The whole set of direction,distance pairs is treated as a series of
translations which are added up to provide the final translation
specifying the intermediate point (relative to the preceding point).

[para]

The last named direction is propagated to the layout system as the
direction to follow. The use of [cmd noturn] is not able to overide
this behaviour.

[para]

At last, the names of the registered directions also serve as
attribute commands, with an implicit attribute [cmd then] in front of
them.

[para]

If no intermediate or last location is specified for the line the
system falls back to a point [var movelength] pixels away from the
starting location, in the current direction as maintained by the
layouting system
