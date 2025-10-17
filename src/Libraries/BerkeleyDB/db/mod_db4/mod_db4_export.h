/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
#ifndef MOD_DB4_EXPORT_H
#define MOD_DB4_EXPORT_H

#include "db_cxx.h"

#if defined(__cplusplus)
extern "C" {
#endif

int mod_db4_db_env_create(DB_ENV **dbenvp, u_int32_t flags);
int mod_db4_db_create(DB **dbp, DB_ENV *dbenv, u_int32_t flags);
void mod_db4_child_clean_request_shutdown();
void mod_db4_child_clean_process_shutdown();

#if defined(__cplusplus)
}
#endif

#endif
