/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#ifndef mach_o_writer_Universal_h
#define mach_o_writer_Universal_h

#include <stdint.h>
#include <mach-o/fat.h>

#include <span>

// mach_o
#include "Architecture.h"
#include "GradedArchitectures.h"
#include "Header.h"
#include "Universal.h"

namespace mach_o {

using namespace mach_o;

/*!
 * @class UniversalWriter
 *
 * @abstract
 *      Abstraction for fat files
 */
struct VIS_HIDDEN UniversalWriter : public Universal
{
    // for building
    static const UniversalWriter*   make(std::span<const Header*>, bool forceFat64=false, bool arm64offEnd=false);

    static const UniversalWriter*   make(std::span<const Slice>, bool forceFat64=false, bool arm64offEnd=false);
    void                            save(char savedPath[PATH_MAX]) const;
    uint64_t                        size() const;
    void                            free() const;   // only called on object allocated by make()
};


} // namespace mach_o

#endif /* mach_o_writer_Universal_h */
