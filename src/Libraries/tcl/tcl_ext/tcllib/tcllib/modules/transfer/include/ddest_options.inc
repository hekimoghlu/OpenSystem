/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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


[opt_def -channel  [arg handle]]

This option specifies that the destination of the data is a channel,
and its associated argument is the handle of the channel to write the
received data to.

[opt_def -file     [arg path]]

This option specifies that the destination of the data is a file, and
its associated argument is the path of the file to write the received
data to.

[opt_def -variable [arg varname]]

This option specifies that the destination of the data is a variable,
and its associated argument contains the name of the variable to write
the received data to. The variable is assumed to be global or
namespaced, anchored at the global namespace.

[opt_def -progress [arg command]]

This option, if specified, defines a command to be invoked for each
chunk of bytes received, allowing the user to monitor the progress of
the reception of the data. The callback is always invoked with one
additional argument, the number of bytes received so far.

