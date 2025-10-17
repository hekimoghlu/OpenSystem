/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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

struct ExceptionEventLocation {
    ExceptionEventLocation() { }
    ExceptionEventLocation(void* stackPosition, const char* functionName, const char* file, unsigned line)
        : stackPosition(stackPosition)
        , functionName(functionName)
        , file(file)
        , line(line)
    { }
    
    void* stackPosition { nullptr }; // Needed for ASAN detect_stack_use_after_return.
    const char* functionName { nullptr };
    const char* file { nullptr };
    unsigned line { 0 };
};

} // namespace JSC

namespace WTF {
    
class PrintStream;

void printInternal(PrintStream&, JSC::ExceptionEventLocation);
    
} // namespace WTF
