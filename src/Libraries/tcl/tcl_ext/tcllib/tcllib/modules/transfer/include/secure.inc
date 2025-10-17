/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

[section {Secure connections}]

One way to secure connections made by objects of this package is to
require the package [package tls] and then configure the option
[option -socketcmd] to force the use of command [cmd tls::socket] to
open the socket.

[example_begin]
    # Load and initialize tls
    package require tls
    tls::init -cafile /path/to/ca/cert -keyfile ...

    # Create a connector with secure socket setup,
    [vset OBJCREATE] -socketcmd tls::socket ...
    ...
[example_end]
