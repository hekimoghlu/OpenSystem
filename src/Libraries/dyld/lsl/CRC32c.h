/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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
#ifndef CRC32c_h
#define CRC32c_h

#include <span>
#include <cstdint>

#include "Defines.h"

namespace lsl {
struct CRC32cImpl;
struct VIS_HIDDEN CRC32c {
    CRC32c();
    CRC32c(CRC32cImpl&);
    operator uint32_t();
    void operator()(uint8_t);
    void operator()(uint16_t);
    void operator()(uint32_t);
    void operator()(uint64_t);
    void operator()(const std::span<std::byte>);
    void reset();
    static CRC32c softwareChecksumer();
    static CRC32c hardwareChecksumer();
private:
    CRC32cImpl& _impl;
    uint32_t    _crc = 0xffffffff;
};
};

#endif /* CRC32c_h */
