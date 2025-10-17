/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#include "config.h"
#include "ConstantValue.h"

#include <wtf/PrintStream.h>
#include <wtf/text/WTFString.h>

namespace WGSL {

ConstantValue ConstantArray::operator[](unsigned index)
{
    return elements[index];
}

ConstantValue ConstantVector::operator[](unsigned index)
{
    return elements[index];
}

ConstantVector ConstantMatrix::operator[](unsigned index)
{
    ConstantVector result(rows);
    for (unsigned i = 0; i < rows; ++i)
        result.elements[i] = elements[index * rows + i];
    return result;
}

void ConstantValue::dump(PrintStream& out) const
{
    WTF::switchOn(*this,
        [&](double d) {
            out.print(String::number(d));
        },
        [&](float f) {
            out.print(String::number(f));
            if (std::isfinite(f))
                out.print("f");
        },
        [&](half h) {
            out.print(String::number(h));
            if (std::isfinite(static_cast<double>(h)))
                out.print("h");
        },
        [&](int64_t i) {
            out.print(String::number(i));
        },
        [&](int32_t i) {
            out.print(String::number(i), "i");
        },
        [&](uint32_t u) {
            out.print(String::number(u), "u");
        },
        [&](bool b) {
            out.print(b ? "true" : "false");
        },
        [&](const ConstantArray& a) {
            out.print("array(");
            bool first = true;
            for (const auto& element : a.elements) {
                if (!first)
                    out.print(", ");
                first = false;
                out.print(element);
            }
            out.print(")");
        },
        [&](const ConstantVector& v) {
            out.print("vec", v.elements.size(), "(");
            bool first = true;
            for (const auto& element : v.elements) {
                if (!first)
                    out.print(", ");
                first = false;
                out.print(element);
            }
            out.print(")");
        },
        [&](const ConstantMatrix& m) {
            out.print("mat", m.columns, "x", m.rows, "(");
            bool first = true;
            for (const auto& element : m.elements) {
                if (!first)
                    out.print(", ");
                first = false;
                out.print(element);
            }
            out.print(")");
        },
        [&](const ConstantStruct& s) {
            out.print("struct { ");
            bool first = true;
            for (const auto& entry : s.fields) {
                if (!first)
                    out.print(", ");
                first = false;
                out.print(entry.key, ": ", entry.value);
            }
            out.print(" }");
        });
}

} // namespace WGSL
