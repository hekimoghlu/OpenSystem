/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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


[para]

If both [cmd width], and [cmd height] are specified then any
specification of [cmd aspect] is ignored, as it is implicitly defined
in the width and height as well, and this takes precedence. A missing
specification is ignored in that case well, i.e. no defaults are
required.

[para]

If the [cmd aspect] is specified, and one of the attributes
[cmd width] or [cmd height], then the missing attribute is calculated
from the two which are specified. No defaults are required for these
cases either.

[para]

If only one of the attributes [cmd width] or [cmd height] is specified
then the system uses a fallback for the [cmd aspect], the value taken
from the language variable [var diamondaspect], which itselfs defaults
to [const 2].

[para]

If none of of the attributes [cmd width] or [cmd height] is specified
then the system uses a fallback for the [cmd width], the value taken
from the language variable [var boxwidth], which itselfs defaults to
the pixel equivalent of [const {2 cm}]. For the aspect it uses either
the user-specified value or the default taken as described in the
previous paragraph.
