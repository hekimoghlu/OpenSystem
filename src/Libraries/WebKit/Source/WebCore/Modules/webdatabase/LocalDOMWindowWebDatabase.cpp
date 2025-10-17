/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#include "LocalDOMWindowWebDatabase.h"

#include "Database.h"
#include "DatabaseManager.h"
#include "Document.h"
#include "LocalDOMWindow.h"
#include "SecurityOrigin.h"

namespace WebCore {

ExceptionOr<RefPtr<Database>> LocalDOMWindowWebDatabase::openDatabase(LocalDOMWindow& window, const String& name, const String& version, const String& displayName, unsigned estimatedSize, RefPtr<DatabaseCallback>&& creationCallback)
{
    if (!window.isCurrentlyDisplayedInFrame())
        return RefPtr<Database> { nullptr };
    auto& manager = DatabaseManager::singleton();
    if (!manager.isAvailable())
        return Exception { ExceptionCode::SecurityError };
    RefPtr document = window.document();
    if (!document)
        return Exception { ExceptionCode::SecurityError };
    document->addConsoleMessage(MessageSource::Storage, MessageLevel::Warning, "Web SQL is deprecated. Please use IndexedDB instead."_s);

    if (document->canAccessResource(ScriptExecutionContext::ResourceType::WebSQL) != ScriptExecutionContext::HasResourceAccess::Yes)
        return Exception { ExceptionCode::SecurityError };

    auto result = manager.openDatabase(*window.document(), name, version, displayName, estimatedSize, WTFMove(creationCallback));
    if (result.hasException()) {
        // FIXME: To preserve our past behavior, this discards the error string in the exception.
        // At a later time we may decide that we want to use the error strings, and if so we can just return the exception as is.
        return Exception { result.releaseException().code() };
    }
    return RefPtr<Database> { result.releaseReturnValue() };
}

} // namespace WebCore
