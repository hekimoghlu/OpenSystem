/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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

namespace JSC {

// FIXME: We should support other modes. see: https://bugs.webkit.org/show_bug.cgi?id=162693
using MemoryModeType = uint8_t;
enum class MemoryMode : MemoryModeType {
    BoundsChecking = 0,
    Signaling,
};

static constexpr size_t numberOfMemoryModes = 2;

enum class MemorySharingMode : uint8_t {
    Default = 0,
    Shared,
};

} // namespace JSC

namespace WTF {

class PrintStream;
JS_EXPORT_PRIVATE void printInternal(PrintStream&, JSC::MemoryMode);
JS_EXPORT_PRIVATE void printInternal(PrintStream&, JSC::MemorySharingMode);

} // namespace WTF
