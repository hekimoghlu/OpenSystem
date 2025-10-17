/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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

#include <wtf/MonotonicTime.h>
#include <wtf/Noncopyable.h>
#include <wtf/text/ASCIILiteral.h>

namespace JSC {

// FIXME: We should find some way of reconciling the differences between WTF::TimingScope and this class. The differences
// are:
// - CompilerTimingScope knows to only do work when --logPhaseTimes=true, while TimingScope is unconditional.
// - CompilerTimingScope reports totals on every run, while TimingScope reports averages periodically.

class CompilerTimingScope {
    WTF_MAKE_NONCOPYABLE(CompilerTimingScope);
public:
    CompilerTimingScope(ASCIILiteral compilerName, ASCIILiteral name);
    ~CompilerTimingScope();

private:
    ASCIILiteral m_compilerName;
    ASCIILiteral m_name;
    MonotonicTime m_before;
};

JS_EXPORT_PRIVATE void logTotalPhaseTimes();

} // namespace JSC
