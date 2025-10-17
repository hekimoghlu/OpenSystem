/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#include "JSCast.h"
#include "Structure.h"
#include <wtf/MonotonicTime.h>
#include <wtf/StackTrace.h>

namespace JSC {

struct CellProfile {
    enum Liveness {
        Unknown,
        Dead,
        Live
    };

    CellProfile(HeapCell* cell, HeapCell::Kind kind, Liveness liveness)
        : m_cell(cell)
        , m_kind(kind)
        , m_liveness(liveness)
        , m_timestamp(MonotonicTime::now())
    {
        if (isJSCellKind(m_kind) && m_liveness != Dead)
            m_className = jsCell()->structure()->classInfoForCells()->className;
    }

    CellProfile(CellProfile&& other)
        : m_cell(other.m_cell)
        , m_kind(other.m_kind)
        , m_liveness(other.m_liveness)
        , m_timestamp(other.m_timestamp)
        , m_className(other.m_className)
        , m_stackTrace(WTFMove(other.m_stackTrace))
    { }

    HeapCell* cell() const { return m_cell; }
    JSCell* jsCell() const
    {
        ASSERT(isJSCell());
        return static_cast<JSCell*>(m_cell);
    }

    bool isJSCell() const { return isJSCellKind(m_kind); }
    
    HeapCell::Kind kind() const { return m_kind; }

    bool isLive() const { return m_liveness == Live; }
    bool isDead() const { return m_liveness == Dead; }

    void setIsLive() { m_liveness = Live; }
    void setIsDead() { m_liveness = Dead; }

    MonotonicTime timestamp() const { return m_timestamp; }

    const char* className() const { return m_className; }

    StackTrace* stackTrace() const { return m_stackTrace.get(); }
    void setStackTrace(StackTrace* trace) { m_stackTrace = std::unique_ptr<StackTrace>(trace); }

private:
    HeapCell* m_cell;
    HeapCell::Kind m_kind;
    Liveness m_liveness { Unknown };
    MonotonicTime m_timestamp;
    const char* m_className { nullptr };
    std::unique_ptr<StackTrace> m_stackTrace;
};

} // namespace JSC
