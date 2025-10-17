/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
/*!
 @header SecItemSchema.h - The thing that does the stuff with the gibli.
 */

#ifndef _SECURITYD_SECITEMSCHEMA_H_
#define _SECURITYD_SECITEMSCHEMA_H_

#include "keychain/securityd/SecDbItem.h"

__BEGIN_DECLS

const SecDbSchema* current_schema(void);
const SecDbSchema * const * all_schemas(void);
// for tests
void set_current_schema_index(int idx);
bool current_schema_index_is_set_for_testing(void);
void reset_current_schema_index(void);

// class accessors for current schema
const SecDbClass* genp_class(void);
const SecDbClass* inet_class(void);
const SecDbClass* cert_class(void);
const SecDbClass* keys_class(void);

// Not really a class per-se
const SecDbClass* identity_class(void);

// Class with 1 element in it which is the database version.
const SecDbClass* tversion_class(void);

const SecDbClass* sharingIncomingQueue_class(void);
const SecDbClass* sharingMirror_class(void);
const SecDbClass* sharingOutgoingQueue_class(void);

// DbUserVersion, from sqlite `PRAGMA user_version`
int32_t getDbUserVersion(SecDbConnectionRef dbconn);
bool setDbUserVersion(int32_t version, SecDbConnectionRef dbconn, CFErrorRef* error);

// For keychain, the user_version is a bit field
typedef enum {
    KeychainDbUserVersion_Transcrypted = 1 << 0, // whether the DB has been transcrypted to be protected by the system keychain keybag. See rdar://94751061
} KeychainDbUserVersion;


// Direct attribute accessors
// If you change one of these, update it here
extern const SecDbAttr v6v_Data;

extern const SecDbAttr v6agrp;
extern const SecDbAttr v6desc;
extern const SecDbAttr v6svce;
extern const SecDbAttr v7vwht;
extern const SecDbAttr v7tkid;
extern const SecDbAttr v7utomb;
extern const SecDbAttr v8musr;
extern const SecDbAttr v10itemuuid;
extern const SecDbAttr v10itempersistentref;

// TODO: Directly expose other important attributes like
// kSecItemSyncAttr, kSecItemTombAttr, kSecItemCdatAttr, kSecItemMdatAttr, kSecItemDataAttr
// This will prevent having to do lookups in SecDbItem for things by kind.

__END_DECLS

#endif /* _SECURITYD_SECITEMSCHEMA_H_ */
