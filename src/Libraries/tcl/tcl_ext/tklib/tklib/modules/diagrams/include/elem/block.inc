/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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

[call [cmd block] [arg script] [arg attr]...]

[include cat_closed.inc]

The main effect is the aggregration of all elements created by the
[arg script] into one element.

This also means that while the elements created by the script are
visible in the element history while the script is executing,
afterward the history contains only the block itself, and not the
elements it is composed of.

[para]

The script has access to the current state of all variables in the
language context.

Any changes to the variables will be reverted after execution of the
block.

However, also, any variables set in the script will be exported as
corners of the element, allowing users to define their own named
locations in the block.

[para]

Regarding the layout mechanism any changes made by the script are
reverted after the element is done.

In other words, a block is an implicit [cmd group].

[para]

Blocks handle all attributes, propgating their settings into the
script as the default values active during script execution.
