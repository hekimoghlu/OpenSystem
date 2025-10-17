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
#ifndef mach_o_writer_SplitSeg_h
#define mach_o_writer_SplitSeg_h

// mach_o
#include "SplitSeg.h"
#include "Error.h"

#include <span>
#include <vector>
#include <unordered_map>

namespace mach_o {

using namespace mach_o;

/*!
 * @class SplitSegInfo
 *
 * @abstract
 *      Class to encapsulate building split seg info
 */
class VIS_HIDDEN SplitSegInfoWriter : public SplitSegInfo
{
public:

                        // used build split seg info
                        // Note: entries so not need to be sorted
                        SplitSegInfoWriter(std::span<const Entry> entries);

    static size_t       estimateSplitSegInfoSize(std::span<const Entry> entries);

    std::span<const uint8_t>  bytes() const { return _bytes; }

private:
    std::vector<uint8_t> _bytes;
    Error                _buildError;
    static const bool    _verbose = false;
};


} // namespace mach_o

#endif // mach_o_writer_CompactUnwind_h
