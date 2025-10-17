/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
[manpage_begin doctools::idx::export::[vset PACKAGE] n [vset VERSION]]
[copyright {2009 Andreas Kupries <andreas_kupries@users.sourceforge.net>}]
[moddesc   {Documentation tools}]
[titledesc "[vset NAME] export plugin"]
[category  {Text formatter plugin}]
[require Tcl 8.4]
[require doctools::idx::export::[vset PACKAGE] [opt [vset VERSION]]]
[include format/[vset REQUIRE].inc]
[keywords doctools index serialization export [vset NAME]]
[description]

This package implements the doctools keyword index export plugin for
the generation of [vset NAME] markup.

[para]

This is an internal package of doctools, for use by the higher level
management packages handling keyword indices, especially [package \
doctools::idx::export], the export manager.

[para]

Using it from a regular interpreter is possible, however only with
contortions, and is not recommended.

The proper way to use this functionality is through the package
[package doctools::idx::export] and the export manager objects it
provides.


[section API]

The API provided by this package satisfies the specification of the
docidx export plugin API version 2.

[list_begin definitions]

[call [cmd export] [arg serial] [arg configuration]]

This command takes the canonical serialization of a keyword index, as
specified in section [sectref {Keyword index serialization format}],
and contained in [arg serial], the [arg configuration], a dictionary,
and generates [vset NAME] markup encoding the index.

The created string is then returned as the result of the command.

[list_end]

[include config/[vset CONFIG].inc]
[include ../serialization.inc]

[vset CATEGORY doctools]
[include ../../../doctools2base/include/feedback.inc]
[manpage_end]
