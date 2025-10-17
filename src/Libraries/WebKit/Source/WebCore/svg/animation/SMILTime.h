/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 26, 2024.
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

#include <algorithm>
#include <cmath>
#include <wtf/Seconds.h>

namespace WebCore {

class SMILTime {
public:
    SMILTime() : m_time(0) { }
    SMILTime(double time) : m_time(time) { ASSERT(!std::isnan(time)); }
    SMILTime(Seconds time) : m_time(time.value()) { ASSERT(!std::isnan(time.value())); }
    SMILTime(const SMILTime& o) : m_time(o.m_time) { }
    
    static SMILTime unresolved() { return unresolvedValue; }
    static SMILTime indefinite() { return indefiniteValue; }
    
    SMILTime& operator=(const SMILTime& o) { m_time = o.m_time; return *this; }
    double value() const { return m_time; }
    
    bool isFinite() const { return m_time < indefiniteValue; }
    bool isIndefinite() const { return m_time == indefiniteValue; }
    bool isUnresolved() const { return m_time == unresolvedValue; }
    
private:
    static const double unresolvedValue;
    static const double indefiniteValue;

    double m_time;
};

class SMILTimeWithOrigin {
public:
    enum Origin {
        ParserOrigin,
        ScriptOrigin
    };

    SMILTimeWithOrigin()
        : m_origin(ParserOrigin)
    {
    }

    SMILTimeWithOrigin(const SMILTime& time, Origin origin)
        : m_time(time)
        , m_origin(origin)
    {
    }

    const SMILTime& time() const { return m_time; }
    bool originIsScript() const { return m_origin == ScriptOrigin; }

private:
    SMILTime m_time;
    Origin m_origin;
};

inline bool operator==(const SMILTime& a, const SMILTime& b) { return a.isFinite() && a.value() == b.value(); }
inline bool operator!(const SMILTime& a) { return !a.isFinite() || !a.value(); }
inline bool operator>(const SMILTime& a, const SMILTime& b) { return a.value() > b.value(); }
inline bool operator<(const SMILTime& a, const SMILTime& b) { return a.value() < b.value(); }
inline bool operator>=(const SMILTime& a, const SMILTime& b) { return a.value() > b.value() || operator==(a, b); }
inline bool operator<=(const SMILTime& a, const SMILTime& b) { return a.value() < b.value() || operator==(a, b); }
inline bool operator<(const SMILTimeWithOrigin& a, const SMILTimeWithOrigin& b) { return a.time() < b.time(); }

SMILTime operator+(const SMILTime&, const SMILTime&);
SMILTime operator-(const SMILTime&, const SMILTime&);
// So multiplying times does not make too much sense but SMIL defines it for duration * repeatCount
SMILTime operator*(const SMILTime&, const SMILTime&);

} // namespace WebCore
