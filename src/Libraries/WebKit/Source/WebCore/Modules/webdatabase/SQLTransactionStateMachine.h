/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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

#include "SQLTransactionState.h"
#include <wtf/Forward.h>

#ifndef NDEBUG
#include <array>
#endif

namespace WebCore {

template<typename T>
class SQLTransactionStateMachine {
public:
    virtual ~SQLTransactionStateMachine() = default;

protected:
    SQLTransactionStateMachine();

    typedef void (T::*StateFunction)();
    virtual StateFunction stateFunctionFor(SQLTransactionState) = 0;

    void setStateToRequestedState();
    void runStateMachine();

    SQLTransactionState m_nextState;
    SQLTransactionState m_requestedState;

#ifndef NDEBUG
    // The state audit trail (i.e. bread crumbs) keeps track of up to the last
    // s_sizeOfStateAuditTrail states that the state machine enters. The audit
    // trail is updated before entering each state. This is for debugging use
    // only.
    static constexpr size_t s_sizeOfStateAuditTrail = 20;
    int m_nextStateAuditEntry;
    std::array<SQLTransactionState, s_sizeOfStateAuditTrail> m_stateAuditTrail;
#endif
};

#if !LOG_DISABLED
extern ASCIILiteral nameForSQLTransactionState(SQLTransactionState);
#endif

template<typename T>
SQLTransactionStateMachine<T>::SQLTransactionStateMachine()
    : m_nextState(SQLTransactionState::Idle)
    , m_requestedState(SQLTransactionState::Idle)
#ifndef NDEBUG
    , m_nextStateAuditEntry(0)
#endif
{
#ifndef NDEBUG
    for (size_t i = 0; i < s_sizeOfStateAuditTrail; ++i)
        m_stateAuditTrail[i] = SQLTransactionState::NumberOfStates;
#endif
}

template<typename T>
void SQLTransactionStateMachine<T>::setStateToRequestedState()
{
    ASSERT(m_nextState == SQLTransactionState::Idle);
    ASSERT(m_requestedState != SQLTransactionState::Idle);
    m_nextState = m_requestedState;
    m_requestedState = SQLTransactionState::Idle;
}

template<typename T>
void SQLTransactionStateMachine<T>::runStateMachine()
{
    ASSERT(SQLTransactionState::End < SQLTransactionState::Idle);

    if (m_nextState <= SQLTransactionState::Idle)
        return;

    ASSERT(m_nextState < SQLTransactionState::NumberOfStates);

    StateFunction stateFunction = stateFunctionFor(m_nextState);
    ASSERT(stateFunction);

#ifndef NDEBUG
    m_stateAuditTrail[m_nextStateAuditEntry] = m_nextState;
    m_nextStateAuditEntry = (m_nextStateAuditEntry + 1) % s_sizeOfStateAuditTrail;
#endif

    (static_cast<T*>(this)->*stateFunction)();
    m_nextState = SQLTransactionState::Idle;
}

} // namespace WebCore
