/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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

[para]

Note here that the Parser Tools are based on Tcl 8.5+. In other words,
the channel argument is not restricted to files, sockets, etc. We have
the full power of [term {reflected channels}] available.

[para]

It should also be noted that the parser pulls the characters from the
input stream as it needs them. If a parser created by this package has
to be operated in a push aka event-driven manner it will be necessary
to go to Tcl 8.6+ and use the [package coroutine::auto] to wrap it
into a coroutine where [cmd read] is properly changed for
push-operation.
