/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
//
//  SDCSPDLDatabase.h - Description t.b.d.
//
#ifndef _H_SD_CSPDLDATABASE
#define _H_SD_CSPDLDATABASE

#include <security_cdsa_plugin/Database.h>
#include <security_cdsa_plugin/DbContext.h>
#include <security_cdsa_utilities/handleobject.h>
#include <security_utilities/refcount.h>
#include <vector>

//
// SDCSPDLDatabaseManager
//
class SDCSPDLDatabaseManager : public DatabaseManager
{
public:
    Database *make(const DbName &inDbName);
};

#endif //_H_SD_CSPDLDATABASE
