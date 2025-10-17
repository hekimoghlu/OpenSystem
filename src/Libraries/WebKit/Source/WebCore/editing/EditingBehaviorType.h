/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#pragma once

namespace WebCore {

// There are multiple editing details that are different on Windows than Macintosh.
// We use a single switch for all of them. Some examples:
//
//    1) Clicking below the last line of an editable area puts the caret at the end
//       of the last line on Mac, but in the middle of the last line on Windows.
//    2) Pushing the down arrow key on the last line puts the caret at the end of the
//       last line on Mac, but does nothing on Windows. A similar case exists on the
//       top line.
//
// This setting is intended to control these sorts of behaviors. There are some other
// behaviors with individual function calls on EditorClient (smart copy and paste and
// selecting the space after a double click) that could be combined with this if
// if possible in the future.
enum class EditingBehaviorType : uint8_t {
    Mac,
    Windows,
    Unix,
    iOS,
};

} // WebCore namespace
