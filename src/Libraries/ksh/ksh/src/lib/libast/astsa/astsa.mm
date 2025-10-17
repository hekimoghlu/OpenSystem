/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

.xx title="astsa"
.MT 4
.TL

.H 1 "astsa"
.B astsa
implements a small subset of the
.B ast
library for other
.B ast
standalone commands and libraries using X/Open interfaces. 
.P
To get better performance and functionality, consider using any of
the full-featured ast-* packages at
.DS
.xx link="http://www.research.att.com/sw/download/"
.DE
.P
astsa.omk is an old make makefile that builds the headers and objects
and defines these variables for use in other makefiles
.VL 12
.LI
.B ASTSA_GEN
point -I to these
.LI
.B ASTSA_HDRS
point -I to these
.LI
.B AST_OBJS
link against these
.LE
The astsa files may be combined in a single directory with other ast
standalone packages.
