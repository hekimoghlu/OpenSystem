/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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

#include "portable.h"
#include "slap.h"
/*
 *  psauth.h
 *  AuthTest
 *
 *  Created by gbv on Mon May 20 2002.
 *  Copyright (c) 2003 Apple Computer, Inc., All rights reserved.
 *
 */

enum {
    kAuthNoError = 0,
    kAuthSASLError = 1,
    kAuthOtherError = 2,
    kAuthKeyError = 3,
    kAuthenticationError = 4
};

#define PASSWORD_SERVER_AUTH_TYPE "ApplePasswordServer"
#define BASIC_AUTH_TYPE "basic"
#define SHADOWHASH_AUTH_TYPE "ShadowHash"

int CheckAuthType(char* inAuthAuthorityData, char* authType);
int DoPSAuth(char* userName, char* password, char* inAuthAuthorityData, Connection *conn, const char *dn);

