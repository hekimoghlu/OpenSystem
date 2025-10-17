/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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
#ifndef __hdb_private_h__
#define __hdb_private_h__

#include <stdarg.h>

krb5_error_code
_hdb_fetch_kvno (
	krb5_context /*context*/,
	HDB */*db*/,
	krb5_const_principal /*principal*/,
	unsigned /*flags*/,
	krb5_kvno /*kvno*/,
	hdb_entry_ex */*entry*/);

hdb_master_key
_hdb_find_master_key (
	int32_t */*mkvno*/,
	hdb_master_key /*mkey*/);

krb5_error_code
_hdb_keytab2hdb_entry (
	krb5_context /*context*/,
	const krb5_keytab_entry */*ktentry*/,
	hdb_entry_ex */*entry*/);

int
_hdb_mkey_decrypt (
	krb5_context /*context*/,
	hdb_master_key /*key*/,
	krb5_key_usage /*usage*/,
	void */*ptr*/,
	size_t /*size*/,
	krb5_data */*res*/);

int
_hdb_mkey_encrypt (
	krb5_context /*context*/,
	hdb_master_key /*key*/,
	krb5_key_usage /*usage*/,
	const void */*ptr*/,
	size_t /*size*/,
	krb5_data */*res*/);

int
_hdb_mkey_version (hdb_master_key /*mkey*/);

krb5_error_code
_hdb_remove (
	krb5_context /*context*/,
	HDB */*db*/,
	krb5_const_principal /*principal*/);

krb5_error_code
_hdb_store (
	krb5_context /*context*/,
	HDB */*db*/,
	unsigned /*flags*/,
	hdb_entry_ex */*entry*/);

#endif /* __hdb_private_h__ */
