/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
[manpage_begin doctools::msgcat::idx::[vset PACKAGE] n 0.1]
[copyright {2009 Andreas Kupries <andreas_kupries@users.sourceforge.net>}]
[moddesc   {Documentation tools}]
[titledesc "Message catalog for the docidx parser ([vset NAME])"]
[category  {Documentation tools}]
[require Tcl 8.4]
[require msgcat]
[require doctools::msgcat::idx::[vset PACKAGE] [opt 0.1]]
[keywords doctools docidx {message catalog}]
[keywords localization l10n internationalization i18n]
[keywords {catalog package} {message package}]
[keywords [vset NAME]]
[description]

The package [package doctools::msgcat::idx::[vset PACKAGE]] is a
support module providing the [vset LONGNAME] language message catalog
for the docidx parser in the doctools system version 2. As such it is
an internal package a regular user (developer) should not be in direct
contact with.

[para]

If you are such please go the documentation of either
[list_begin enumerated]
[enum] [package doctools::doc],
[enum] [package doctools::toc], or
[enum] [package doctools::idx]
[list_end]
[para]

Within the system architecture this package resides under the package
[package doctools::msgcat] providing the general message catalog
management within the system. [emph Note] that there is [emph no]
explicit dependency between the manager and catalog packages. The
catalog is a plugin which is selected and loaded dynamically.


[section API]

This package has no exported API.


[vset CATEGORY doctools]
[include ../../doctools2base/include/feedback.inc]
[manpage_end]
