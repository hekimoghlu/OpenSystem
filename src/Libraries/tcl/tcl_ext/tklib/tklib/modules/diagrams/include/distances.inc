/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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

[call [arg number] [const cm]]
[call [arg number] [const mm]]
[call [arg number] [const inch]]
[call [arg number] [const pt]]

These commands allow the specification of distances and coordinates in
metric and imperial units, returning the equivalent distance or
coordinate in pixels, which is the unit used internally for all
calculations.

[para]

The conversion factors are based on the result of [cmd {tk scaling}]
and are computed once, at the time the package is sourced, future
changes of the [cmd {tk scaling}] factor have no effect.
