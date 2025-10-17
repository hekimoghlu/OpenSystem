/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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

[comment {-*- tcl -*- doctools = digest_header.inc}]
[include common/trf_version.inc]
[manpage_begin [vset digest] n [vset trf_version]]
[titledesc "Message digest \"[vset digest]\""]
[include common/trf_header.inc]
[description]

The command [cmd [vset digest]] is one of several message digests
provided by the package [package trf]. See [syscmd trf-intro] for an
overview of the whole package.

[para]

[list_begin definitions]

[call [cmd [vset digest]] [opt [arg options...]] [opt [arg data]]]

The options listed below are understood by the digest if and only if
the digest is [term attached] to a channel.

See section [sectref {IMMEDIATE versus ATTACHED}] for an explanation
of the term [term attached].

[list_begin definitions]
[include digest/options.inc]
[list_end]
[nl]

The options listed below are always understood by the digest,

[term attached] versus [term immediate] does not matter. See section
[sectref {IMMEDIATE versus ATTACHED}] for explanations of these two
terms.


[list_begin definitions]
[include common/options.inc]
[list_end]
[list_end]
