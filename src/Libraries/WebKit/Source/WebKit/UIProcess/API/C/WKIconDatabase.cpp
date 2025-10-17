/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
#include "WKIconDatabase.h"

#include "APIData.h"
#include "WKAPICast.h"
#include "WebIconDatabase.h"

using namespace WebKit;

WKTypeID WKIconDatabaseGetTypeID()
{
    return toAPI(WebIconDatabase::APIType);
}

void WKIconDatabaseSetIconDatabaseClient(WKIconDatabaseRef, const WKIconDatabaseClientBase*)
{
}

void WKIconDatabaseRetainIconForURL(WKIconDatabaseRef, WKURLRef)
{
}

void WKIconDatabaseReleaseIconForURL(WKIconDatabaseRef, WKURLRef)
{
}

void WKIconDatabaseSetIconDataForIconURL(WKIconDatabaseRef, WKDataRef, WKURLRef)
{
}

void WKIconDatabaseSetIconURLForPageURL(WKIconDatabaseRef, WKURLRef, WKURLRef)
{
}

WKURLRef WKIconDatabaseCopyIconURLForPageURL(WKIconDatabaseRef, WKURLRef)
{
    return nullptr;
}

WKDataRef WKIconDatabaseCopyIconDataForPageURL(WKIconDatabaseRef, WKURLRef)
{
    return nullptr;
}

void WKIconDatabaseEnableDatabaseCleanup(WKIconDatabaseRef)
{
}

void WKIconDatabaseRemoveAllIcons(WKIconDatabaseRef)
{
}

void WKIconDatabaseCheckIntegrityBeforeOpening(WKIconDatabaseRef)
{
}

void WKIconDatabaseClose(WKIconDatabaseRef)
{
}
