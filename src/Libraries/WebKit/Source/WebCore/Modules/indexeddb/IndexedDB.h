/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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

namespace WebCore {

namespace IndexedDB {

enum class TransactionState : uint8_t {
    Active,
    Inactive,
    Committing,
    Aborting,
    Finished,
};

enum class CursorDirection : uint8_t {
    Next,
    Nextunique,
    Prev,
    Prevunique,
};
const unsigned CursorDirectionMaximum = 3;

enum class CursorType : bool {
    KeyAndValue,
    KeyOnly,
};
const unsigned CursorTypeMaximum = 1;

enum class CursorSource : bool {
    Index,
    ObjectStore,
};

enum class VersionNullness : uint8_t {
    Null,
    NonNull,
};

enum class ObjectStoreOverwriteMode : uint8_t {
    Overwrite,
    OverwriteForCursor,
    NoOverwrite,
};

enum class IndexRecordType : bool {
    Key,
    Value,
};

enum class ObjectStoreRecordType : uint8_t {
    ValueOnly,
    KeyOnly,
};

// In order of the least to the highest precedent in terms of sort order.
enum class KeyType : int8_t {
    Max = -1,
    Invalid = 0,
    Array,
    Binary,
    String,
    Date,
    Number,
    Min,
};

enum class RequestType : uint8_t {
    Open,
    Delete,
    Other,
};

enum class GetAllType : bool {
    Keys,
    Values,
};

enum class ConnectionClosedOnBehalfOfServer : bool { No, Yes };

enum class CursorIterateOption : bool {
    DoNotReply,
    Reply,
};

} // namespace IndexedDB

} // namespace WebCore
