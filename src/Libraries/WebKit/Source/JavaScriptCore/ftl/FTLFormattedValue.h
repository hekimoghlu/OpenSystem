/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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

#if ENABLE(FTL_JIT)

#include "DataFormat.h"
#include "FTLAbbreviatedTypes.h"

namespace JSC { namespace FTL {

// This class is mostly used for OSR; it's a way of specifying how a value is formatted
// in cases where it wouldn't have been obvious from looking at other indicators (like
// the type of the B3::Value* or the type of the DFG::Node).

class FormattedValue {
public:
    FormattedValue()
        : m_format(DataFormatNone)
        , m_value(nullptr)
    {
    }
    
    FormattedValue(DataFormat format, LValue value)
        : m_format(format)
        , m_value(value)
    {
    }
    
    bool operator!() const
    {
        ASSERT((m_format == DataFormatNone) == !m_value);
        return m_format == DataFormatNone;
    }
    
    DataFormat format() const { return m_format; }
    LValue value() const { return m_value; }

private:
    DataFormat m_format;
    LValue m_value;
};

static inline FormattedValue noValue() { return FormattedValue(); }
static inline FormattedValue int32Value(LValue value) { return FormattedValue(DataFormatInt32, value); }
static inline FormattedValue booleanValue(LValue value) { return FormattedValue(DataFormatBoolean, value); }
static inline FormattedValue jsValueValue(LValue value) { return FormattedValue(DataFormatJS, value); }
static inline FormattedValue doubleValue(LValue value) { return FormattedValue(DataFormatDouble, value); }

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
