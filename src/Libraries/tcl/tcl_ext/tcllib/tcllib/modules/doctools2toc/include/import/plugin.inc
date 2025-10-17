/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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

[comment {-*- tcl -*- doctools manpage}]
[manpage_begin doctools::toc::import::[vset PACKAGE] n 0.1]
[copyright {2009 Andreas Kupries <andreas_kupries@users.sourceforge.net>}]
[moddesc   {Documentation tools}]
[titledesc "[vset NAME] import plugin"]
[category  {Text formatter plugin}]
[require Tcl 8.4]
[require doctools::toc::import::[vset PACKAGE] [opt 0.1]]
[include format/[vset REQUIRE].inc]
[keywords doctools toc {table of contents} deserialization import [vset NAME]]
[description]

This package implements the doctools table of contents import plugin
for the parsing of [vset NAME] markup.

[para]

This is an internal package of doctools, for use by the higher level
management packages handling tables of contents, especially [package \
doctools::toc::import], the import manager.

[para]

Using it from a regular interpreter is possible, however only with
contortions, and is not recommended.

The proper way to use this functionality is through the package
[package doctools::toc::import] and the import manager objects it
provides.


[section API]

The API provided by this package satisfies the specification of the
doctoc import plugin API version 2.

[list_begin definitions]

[call [cmd import] [arg string] [arg configuration]]

This command takes the [arg string] and parses it as [vset NAME]
markup encoding a table of contents, in the context of the specified
[arg configuration] (a dictionary). The result of the command is the
canonical serialization of that table of contents, in the form
specified in section [sectref {ToC serialization format}].

[list_end]


[include config/[vset CONFIG].inc]
[include ../serialization.inc]

[vset CATEGORY doctools]
[include ../../../doctools2base/include/feedback.inc]
[manpage_end]
