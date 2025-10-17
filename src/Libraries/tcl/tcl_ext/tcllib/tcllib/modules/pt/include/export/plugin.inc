/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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

[comment {-*- tcl -*- --- !doctools ---}]
[manpage_begin pt::peg::export::[vset PACKAGE] n [vset VERSION]]
[include ../module.inc]
[include ../keywords_export.inc]
[titledesc "PEG Export Plugin. Write [vset NAME] format"]
[require pt::peg::export::[vset PACKAGE] [opt [vset VERSION]]]
[require pt::peg::to::[vset PACKAGE]]
[description]
[include ../ref_intro.inc]

This package implements the parsing expression grammar export plugin
for the generation of [vset NAME] markup.

[para]

It resides in the Export section of the Core Layer of Parser Tools and
is intended to be used by [package pt::peg::export], the export
manager, sitting between it and the corresponding core conversion
functionality provided by [package pt::peg::to::[vset PACKAGE]].

[para][image arch_core_eplugins][para]

[para]

While the direct use of this package with a regular interpreter is
possible, this is strongly disrecommended and requires a number of
contortions to provide the expected environment.

The proper way to use this functionality depends on the situation:

[list_begin enumerated]
[enum]

In an untrusted environment the proper access is through the package
[package pt::peg::export] and the export manager objects it
provides.

[enum]

In a trusted environment however simply use the package
[package pt::peg::to::[vset PACKAGE]] and access the core
conversion functionality directly.

[list_end]


[section API]

The API provided by this package satisfies the specification of the
Plugin API found in the [manpage {Parser Tools Export API}]
specification.

[list_begin definitions]

[call [cmd export] [arg serial] [arg configuration]]

This command takes the canonical serialization of a parsing expression
grammar, as specified in section [sectref {PEG serialization format}],
and contained in [arg serial], the [arg configuration], a dictionary,
and generates [vset NAME] markup encoding the grammar.

The created string is then returned as the result of the command.

[list_end]

[include config/[vset CONFIG].inc]
[include ../format/[vset PACKAGE].inc]
[include ../serial/pegrammar.inc]
[include ../serial/pexpression.inc]
[include ../feedback.inc]
[manpage_end]
