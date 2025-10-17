/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#ifndef mach_o_writer_BindOpcodes_h
#define mach_o_writer_BindOpcodes_h

#include <stdint.h>
#include <stdio.h>
#include <optional>
#include <span>
#include <vector>

// mach_o
#include "BindOpcodes.h"

namespace mach_o {

using namespace mach_o;

/*!
 * @class BindOpcodes
 *
 * @abstract
 *      Class to encapsulate building bind opcodes
 */
class VIS_HIDDEN BindOpcodesWriter : public BindOpcodes
{
public:
    // Note 'binds' input will be sorted by this method
    BindOpcodesWriter(std::span<LocAndTarget> binds, bool is64);

private:
    std::vector<uint8_t> _opcodes;
};

class VIS_HIDDEN LazyBindOpcodesWriter : public LazyBindOpcodes
{
public:
    // used by unit tests to build opcodes
    typedef void (^LazyStartRecorder)(size_t offset, const char* symbolName);

    // used by unit tests to build opcodes
    LazyBindOpcodesWriter(std::span<LocAndTarget> binds, bool is64, LazyStartRecorder recorder);

private:
    std::vector<uint8_t> _opcodes;
};

class VIS_HIDDEN WeakBindOpcodesWriter : public WeakBindOpcodes
{
public:
    // used by unit tests to build opcodes
    WeakBindOpcodesWriter(std::span<LocAndTarget> binds, bool is64);

private:
    std::vector<uint8_t> _opcodes;
};


} // namespace mach_o

#endif // mach_o_writer_BindOpcodes_h


