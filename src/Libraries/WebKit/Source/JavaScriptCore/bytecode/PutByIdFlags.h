/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

#include "ECMAMode.h"

namespace JSC {

class PutByIdFlags {
public:
    constexpr static PutByIdFlags create(ECMAMode ecmaMode)
    {
        return PutByIdFlags(false, ecmaMode);
    }

    // A direct put_by_id means that we store the property without checking if the
    // prototype chain has a setter.
    constexpr static PutByIdFlags createDirect(ECMAMode ecmaMode)
    {
        return PutByIdFlags(true, ecmaMode);
    }

    bool isDirect() const { return m_isDirect; }
    ECMAMode ecmaMode() const { return m_ecmaMode; }

private:
    constexpr PutByIdFlags(bool isDirect, ECMAMode ecmaMode)
        : m_isDirect(isDirect)
        , m_ecmaMode(ecmaMode)
    {
    }

    bool m_isDirect;
    ECMAMode m_ecmaMode;
};

} // namespace JSC

namespace WTF {

class PrintStream;

void printInternal(PrintStream&, JSC::PutByIdFlags);

} // namespace WTF
