/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
#include "SQLTransactionStateMachine.h"

#include "Logging.h"
#include <wtf/Assertions.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebCore {

#if !LOG_DISABLED
ASCIILiteral nameForSQLTransactionState(SQLTransactionState state)
{
    switch (state) {
    case SQLTransactionState::End:
        return "end"_s;
    case SQLTransactionState::Idle:
        return "idle"_s;
    case SQLTransactionState::AcquireLock:
        return "acquireLock"_s;
    case SQLTransactionState::OpenTransactionAndPreflight:
        return "openTransactionAndPreflight"_s;
    case SQLTransactionState::RunStatements:
        return "runStatements"_s;
    case SQLTransactionState::PostflightAndCommit:
        return "postflightAndCommit"_s;
    case SQLTransactionState::CleanupAndTerminate:
        return "cleanupAndTerminate"_s;
    case SQLTransactionState::CleanupAfterTransactionErrorCallback:
        return "cleanupAfterTransactionErrorCallback"_s;
    case SQLTransactionState::DeliverTransactionCallback:
        return "deliverTransactionCallback"_s;
    case SQLTransactionState::DeliverTransactionErrorCallback:
        return "deliverTransactionErrorCallback"_s;
    case SQLTransactionState::DeliverStatementCallback:
        return "deliverStatementCallback"_s;
    case SQLTransactionState::DeliverQuotaIncreaseCallback:
        return "deliverQuotaIncreaseCallback"_s;
    case SQLTransactionState::DeliverSuccessCallback:
        return "deliverSuccessCallback"_s;
    default:
        return "UNKNOWN"_s;
    }
}
#endif

} // namespace WebCore
