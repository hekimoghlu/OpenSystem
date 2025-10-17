/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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

#include "SQLTransaction.h"
#include <wtf/Forward.h>

namespace WebCore {

class SQLError;

class ChangeVersionWrapper : public SQLTransactionWrapper {
public:
    static Ref<ChangeVersionWrapper> create(String&& oldVersion, String&& newVersion) { return adoptRef(*new ChangeVersionWrapper(WTFMove(oldVersion), WTFMove(newVersion))); }

    bool performPreflight(SQLTransaction&) override;
    bool performPostflight(SQLTransaction&) override;
    SQLError* sqlError() const override { return m_sqlError.get(); };
    void handleCommitFailedAfterPostflight(SQLTransaction&) override;

private:
    ChangeVersionWrapper(String&& oldVersion, String&& newVersion);

    String m_oldVersion;
    String m_newVersion;
    RefPtr<SQLError> m_sqlError;
};

} // namespace WebCore
