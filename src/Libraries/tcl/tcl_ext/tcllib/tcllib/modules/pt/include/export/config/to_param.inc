/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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


[section "[vset TPREFIX]Options"]

The converter to PARAM markup recognizes the following configuration
variables and changes its behaviour as they specify.

[list_begin options]

[opt_def -template string]

The value of this configuration variable is a string into which to put
the generated text and the other configuration settings. The various
locations for user-data are expected to be specified with the
placeholders listed below. The default value is "[const @code@]".

[list_begin definitions]

[def [const @user@]]
To be replaced with the value of the configuration variable [option -user].

[def [const @format@]]
To be replaced with the the constant [const PARAM].

[def [const @file@]]
To be replaced with the value of the configuration variable [option -file].

[def [const @name@]]
To be replaced with the value of the configuration variable [option -name].

[def [const @code@]]
To be replaced with the generated text.

[list_end]

[opt_def -name string]

The value of this configuration variable is the name of the grammar
for which the conversion is run. The default value is [const a_pe_grammar].

[opt_def -user string]

The value of this configuration variable is the name of the user for
which the conversion is run. The default value is [const unknown].

[opt_def -file string]

The value of this configuration variable is the name of the file or
other entity from which the grammar came, for which the conversion is
run. The default value is [const unknown].

[list_end]
