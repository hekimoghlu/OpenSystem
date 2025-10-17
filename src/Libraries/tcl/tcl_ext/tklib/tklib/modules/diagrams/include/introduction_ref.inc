/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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


Welcome to [package diagram], a package for the easy construction of
diagrams (sic), i.e. 2D vector graphics, sometimes also called [term \
pictures].

Note that this package is not a replacement for [package Tk]'s canvas,
but rather a layer sitting on top of it, to make it easier to use.

In other words, using the canvas as the core graphics engine [package \
diagram] abstracts away from the minutiae of handling coordinates to
position and size the drawn elements, allowing the user to concentrate
on the content of the diagram instead.

[para]

This is similar to Brian Kernighan's PIC language for troff, which is
the spiritual ancestor of this package.

[para]

This document contains the reference to the API and drawing (language)
commands. Its intended audience are users of the package wishing to
refresh their memory.

Newcomers should read the [term {Diagram Language Tutorial}] first.

Developers wishing to work on the internals of the package and its
supporting packages should look at section
	[sectref {Diagram Classes}]
first, and then the comments in the sources of the packages itself.

[para]

In the remainder of the document we first describe the APIs of the
diagram class and its instances, followed by the language reference
for the drawing language itself.
